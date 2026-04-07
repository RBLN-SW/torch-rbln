import ctypes
import os
import warnings
from importlib.metadata import PackageNotFoundError, version
from typing import Any  # noqa: UP035

from torch_rbln._internal.env_utils import is_diagnose_mode
from torch_rbln._internal.profiling import (
    emit_rbln_overhead_summary,
    format_rbln_overhead_summary,
    get_rbln_overhead_profile_snapshot,
    has_rbln_overhead_profile_samples,
    is_rbln_overhead_profiling_enabled,
    log_rbln_overhead_summary,
    maybe_emit_rbln_overhead_summary,
    reset_rbln_overhead_profile,
)
from torch_rbln._internal.tvm_libinfo import get_dll_directories, get_dll_directory_candidates


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
        import torch

        # Load shared objects ##################################################
        global library_names, libraries

        # Ensure dependent shared library is loaded before importing native extension
        find_and_load_tvm_library("librbln.so")

        # Import native extension module (e.g., torch_rbln.so)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        for library_name in library_names:
            library_path = os.path.join(current_dir, "lib", library_name)
            try:
                libraries.append(ctypes.CDLL(library_path))
            except OSError as e:
                raise ImportError(
                    f"Failed to load required RBLN shared library `{library_name}` from `{library_path}`. "
                    "Run `python -m torch_rbln.diagnose` for environment diagnostics."
                ) from e

        # Configure RBLN backend ###############################################
        torch.utils.rename_privateuse1_backend("rbln")

        # Importing this occur dlopen("_C.so"). This have to be called after dlopen("libc10_rbln.so").
        import torch_rbln.device

        torch._register_device_module("rbln", torch_rbln.device)

        # Import operators #####################################################
        import torch_rbln._internal.register_ops

        # Apply monkey patches for RBLN functionality ###########################
        from torch_rbln._internal.monkey_patches import apply_all_patches

        apply_all_patches()

        # Set global dynamo configuration ######################################
        # NOTE: RBLN eager mode uses torch.dynamo; a full cache triggers a user-visible warning.
        # Use a slightly larger limit so normal workloads stay below it; prefer an explicit
        # suppression hook if Dynamo exposes one in the future.
        torch._dynamo.config.cache_size_limit = 64

        # TODO: explore Dynamo/compiler options to embed scalars in graphs for reuse (specialize_float is related).
        torch._dynamo.config.specialize_float = True

        # NOTE:
        # This enables hard failure (exception) when either the per-frame `recompile_limit`
        # or the global `accumulated_recompile_limit` is exceeded.
        # Required for the `except torch._dynamo.exc.FailOnRecompileLimitHit` block to trigger.
        torch._dynamo.config.fail_on_cache_limit_hit = True

        # Note: torch.compile monkey patch is applied in apply_all_patches() above

        # Initialize distributed support #######################################
        _initialize_distributed_bindings()
    except Exception:
        status = "uninitialized"
        raise

    # Finalize initialization ##################################################
    status = "initialized"


def find_and_load_tvm_library(target_name: str) -> None:
    """
    Search for a shared library (.so) file in known tvm lib paths and load it with ctypes.

    This function iterates through directories listed by 'tvm_libinfo'. If the target
    shared library is found, it attempts to load it using `ctypes.CDLL`.

    Args:
        target_name (str): The name of the shared library to find and load (e.g., 'librbln.so').

    Raises:
        FileNotFoundError: If the library is not found in any known site-packages paths.
        OSError: If the library is found but fails to load via ctypes.
    """
    search_paths = get_dll_directories()
    for base in search_paths:
        for root, _, files in os.walk(base):
            if target_name in files:
                so_path = os.path.join(root, target_name)
                try:
                    ctypes.CDLL(so_path, ctypes.RTLD_GLOBAL)
                    return
                except OSError as e:
                    raise OSError(f"Failed to load {target_name} at {so_path}: {e}") from e

    if target_name == "librbln.so":
        candidates = get_dll_directory_candidates()
        searched = [p for p, _ in candidates]
        diag_note = (
            " Run 'python -m torch_rbln.diagnose' (or "
            "'TORCH_RBLN_DIAGNOSE=1 python -m torch_rbln.diagnose' if import fails) "
            "for full environment diagnostics."
        )
        env_hint = []
        for var in ("REBEL_HOME", "LD_LIBRARY_PATH", "PYTHONPATH"):
            v = os.environ.get(var, "")
            if v:
                env_hint.append(f"{var}={v[:80]}{'...' if len(v) > 80 else ''}")
        if env_hint:
            env_str = "; ".join(env_hint)
            diag_note += f" Relevant env: {env_str}."
        raise FileNotFoundError(
            "Could not find librbln.so. "
            "Searched directories (in order): "
            + ", ".join(searched[:10])
            + (" ..." if len(searched) > 10 else "")
            + ". "
            "Please install the REBEL compiler (rebel-compiler package) or fix REBEL_HOME/LD_LIBRARY_PATH/PYTHONPATH."
            + diag_note
        )
    raise FileNotFoundError(f"{target_name} not found in any known site-packages path.")


def _create_process_group_rbln(dist_backend_opts, pg_options):
    """
    Create a ProcessGroupRBLN instance for distributed training.

    This function is used as a factory for creating ProcessGroupRBLN instances
    when the RBLN backend is registered with PyTorch's distributed system.
    When available, a Gloo backend is created and passed for non-float16
    allreduce/reduce_scatter fallback.

    Args:
        dist_backend_opts: Distributed backend options containing store, rank, size, timeout
        pg_options: Process group options (unused for RBLN)

    Returns:
        ProcessGroupRBLN: A new ProcessGroupRBLN instance
    """
    import torch_rbln._C

    # Extract parameters from dist_backend_opts
    store = dist_backend_opts.store
    group_rank = dist_backend_opts.group_rank
    group_size = dist_backend_opts.group_size
    group_id = int(dist_backend_opts.group_id)
    global_ranks_in_group = dist_backend_opts.global_ranks_in_group
    timeout = dist_backend_opts.timeout

    # Create Gloo backend for non-float16 allreduce/reduce_scatter fallback when available
    gloo_backend = None
    try:
        from torch.distributed.distributed_c10d import is_gloo_available, ProcessGroupGloo

        if is_gloo_available():
            gloo_backend = ProcessGroupGloo(store, group_rank, group_size, timeout=timeout)
    except Exception:
        pass

    return torch_rbln._C._distributed_c10d.ProcessGroupRBLN(
        store, group_rank, group_size, group_id, global_ranks_in_group, timeout, gloo_backend=gloo_backend
    )


def _initialize_distributed_bindings() -> None:
    """
    Initialize distributed c10d bindings for RBLN.

    This function ensures that the ProcessGroupRBLN Python bindings are
    properly initialized before they are used.

    """
    try:
        import torch_rbln._C

        torch_rbln._C._c10d_rbln_init()
    except Exception as e:
        warnings.warn(f"Failed to initialize distributed c10d bindings: {e}", stacklevel=2)

    _register_distributed_backend_for_rbln()


def _register_distributed_backend_for_rbln() -> None:
    """
    Register the RBLN distributed backend with PyTorch.

    This function registers the RBLN backend so that it can be used with
    torch.distributed.init_process_group(backend='rbln-ccl').
    """
    import torch.distributed as dist

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
