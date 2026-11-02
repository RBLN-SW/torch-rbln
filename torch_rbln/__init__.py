import atexit
import contextlib
import ctypes
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from torch_rbln._internal.abi_check import check_librbln_abi
from torch_rbln._internal.env_utils import is_diagnose_mode
from torch_rbln._internal.rbln_runtime_lib import load_runtime_library
from torch_rbln._internal.rdma_env import _apply_control_plane_ips


try:
    __version__ = version("torch-rbln")
except PackageNotFoundError:
    __version__ = "0.0.0"  # fallback for editable/dev installs without metadata

library_names: list[str] = ["libc10_rbln.so", "libtorch_rbln.so"]
libraries: list[ctypes.CDLL] = []
status: str = "uninitialized"


def torch_backends_entry_point() -> None:
    # Begin initialization #####################################################
    # For once call
    global status
    if status != "uninitialized":
        return
    status = "initializing"
    try:
        # Import torch early so libtorch.so / libc10.so are loaded before our
        # native extensions (which link against them).  When the wheel is built
        # with build-isolation the RUNPATH baked into our .so files points to a
        # temporary directory that no longer exists at install time; pre-loading
        # torch ensures the dynamic linker can resolve the symbols anyway.
        import torch  # noqa: PLC0415

        # Load shared objects ##################################################
        # Map librbln.so before the native extensions: they declare it NEEDED by SONAME,
        # so the loader reuses this mapping instead of searching a RUNPATH baked in at
        # build time.
        librbln_path = load_runtime_library()

        # Verify the rebel ABI contract while librbln.so is the only rebel code loaded.
        # Past this point our extensions bind to its entry points and a mismatch stops
        # being reportable: CPython opens them RTLD_NOW, so a missing symbol aborts the
        # import as `undefined symbol`. It re-opens the mapping just made RTLD_NOLOAD,
        # never a copy, and owns that step so a handle it cannot take fails open like
        # the rest of the check.
        check_librbln_abi(librbln_path)

        # Import native extension module (e.g., torch_rbln.so)
        lib_dir = Path(__file__).absolute().parent / "lib"
        for library_name in library_names:
            library_path = str(lib_dir / library_name)
            try:
                libraries.append(ctypes.CDLL(library_path))
            except OSError as e:
                msg = (
                    f"Failed to load required RBLN shared library `{library_name}` from `{library_path}`. "
                    "Run `python -m torch_rbln.diagnose` for environment diagnostics."
                )
                raise ImportError(msg) from e

        # Configure RBLN backend ###############################################
        torch.utils.rename_privateuse1_backend("rbln")

        # Importing this occur dlopen("_C.so"). This have to be called after
        # dlopen("libc10_rbln.so").
        import torch_rbln.device  # noqa: PLC0415

        torch._register_device_module("rbln", torch_rbln.device)

        # At interpreter teardown, mark the runtime as shutting down so late frees /
        # best-effort ops stop dispatching into a possibly-unmapped runtime (which
        # would SEGFAULT). Best-effort defense; mirrors CUDA teardown handling.
        import torch_rbln._C  # noqa: PLC0415

        atexit.register(torch_rbln._C._set_runtime_shutting_down, shutting_down=True)

        # Import operators #####################################################
        import torch_rbln._internal.register_ops  # noqa: PLC0415

        # Apply monkey patches for RBLN functionality ###########################
        from torch_rbln._internal.monkey_patches import apply_all_patches  # noqa: PLC0415

        apply_all_patches()

        # Set global dynamo configuration ######################################
        # NOTE: RBLN eager mode uses torch.dynamo; a full cache triggers a user-visible
        # warning. Use a slightly larger limit so normal workloads stay below it; prefer
        # an explicit suppression hook if Dynamo exposes one in the future.
        torch._dynamo.config.cache_size_limit = 64

        # TODO: explore Dynamo/compiler options to embed scalars in graphs for reuse
        # (specialize_float is related).
        torch._dynamo.config.specialize_float = True  # pyrefly: ignore[bad-assignment]

        # NOTE:
        # This enables hard failure (exception) when either the per-frame
        # `recompile_limit` or the global `accumulated_recompile_limit` is exceeded.
        # Required for the `except torch._dynamo.exc.FailOnRecompileLimitHit` block to
        # trigger.
        torch._dynamo.config.fail_on_cache_limit_hit = True

        # Note: torch.compile monkey patch is applied in apply_all_patches() above

        # Register torch.profiler (kineto) bridge ##############################
        _initialize_kineto_profiler()

        # Initialize distributed support #######################################
        _initialize_distributed_bindings()
    except Exception:
        status = "uninitialized"
        raise

    # Finalize initialization ##################################################
    status = "initialized"


def _create_process_group_rbln(dist_backend_opts, pg_options):  # noqa: ARG001
    """Create a ProcessGroupRBLN instance for distributed training.

    This function is used as a factory for creating ProcessGroupRBLN instances
    when the RBLN backend is registered with PyTorch's distributed system.
    When available, a Gloo backend is created and passed for non-float16
    allreduce/reduce_scatter fallback.

    Args:
        dist_backend_opts: Distributed backend options containing store, rank,
            size, timeout
        pg_options: Process group options (unused for RBLN)

    Returns:
        ProcessGroupRBLN: A new ProcessGroupRBLN instance
    """
    import torch_rbln._C  # noqa: PLC0415

    _apply_control_plane_ips()

    # Extract parameters from dist_backend_opts
    store = dist_backend_opts.store
    group_rank = dist_backend_opts.group_rank
    group_size = dist_backend_opts.group_size
    group_id = int(dist_backend_opts.group_id)
    global_ranks_in_group = dist_backend_opts.global_ranks_in_group
    timeout = dist_backend_opts.timeout

    # Create Gloo backend for non-float16 allreduce/reduce_scatter fallback when
    # available
    gloo_backend = None
    with contextlib.suppress(Exception):
        from torch.distributed.distributed_c10d import is_gloo_available, ProcessGroupGloo  # noqa: PLC0415

        if is_gloo_available():
            gloo_backend = ProcessGroupGloo(store, group_rank, group_size, timeout=timeout)

    return torch_rbln._C._distributed_c10d.ProcessGroupRBLN(
        store, group_rank, group_size, group_id, global_ranks_in_group, timeout, gloo_backend=gloo_backend
    )


def _initialize_kineto_profiler() -> None:
    """Register the rbln torch.profiler (kineto) bridge (a runtime-free libkineto
    factory registration).

    Do NOT query the device arch here (e.g. ``is_atom_device()``): ``get_npu_name``
    resolves a device, opening and closing a device node on every import. ATOM is gated
    by the runtime instead: ``rbln_kineto_is_active()`` (which the C++ profiler
    ``configure()`` checks) reports inactive on ATOM (rebel-compiler #12079), so no rbln
    session is created there.
    """
    try:
        import torch_rbln._C  # noqa: PLC0415

        torch_rbln._C._register_kineto_profiler()
    except Exception as e:
        warnings.warn(f"Failed to register rbln kineto profiler: {e}", stacklevel=2)


def _initialize_distributed_bindings() -> None:
    """Initialize distributed c10d bindings for RBLN.

    This function ensures that the ProcessGroupRBLN Python bindings are
    properly initialized before they are used.

    """
    try:
        import torch_rbln._C  # noqa: PLC0415

        torch_rbln._C._c10d_rbln_init()
    except Exception as e:
        warnings.warn(f"Failed to initialize distributed c10d bindings: {e}", stacklevel=2)

    _register_distributed_backend_for_rbln()


def _register_distributed_backend_for_rbln() -> None:
    """Register the RBLN distributed backend with PyTorch.

    This function registers the RBLN backend so that it can be used with
    torch.distributed.init_process_group(backend='rbln-ccl').
    """
    import torch.distributed as dist  # noqa: PLC0415

    try:
        dist.Backend.register_backend(
            "rbln-ccl",
            lambda dist_backend_opts, pg_options: _create_process_group_rbln(dist_backend_opts, pg_options),
            extended_api=True,
            devices=["rbln", "cpu"],
        )
    except RuntimeError as e:
        if "already registered" in str(e):
            # Backend is already registered, which is fine
            pass
        else:
            warnings.warn(f"Failed to register RBLN backend: {e}", stacklevel=2)


# Initialize the torch-rbln package (skip when running diagnostics only)
if not is_diagnose_mode():
    torch_backends_entry_point()
