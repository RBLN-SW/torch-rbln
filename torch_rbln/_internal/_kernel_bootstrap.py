"""Bootstrap module for the C++ hot-path kernels.

C++ kernels invoke ``build_*_runtime`` helpers here on cache miss to obtain
a rebel ``DynamoRuntime`` whose ``_runtime_handle`` is a ``PyRblnSyncRuntime``.
The C++ side then caches the runtime and drives ``PyRblnSyncRuntime::Run()``
directly on subsequent hits, bypassing the Python slow path entirely.

Raw-pointer extraction: rebel's ``_C`` module is compiled against pybind11 2.x
(``__pybind11_internals_v4_...``) while torch (and therefore torch-rbln) uses
pybind11 3.x (``__pybind11_internals_v11_...``). Their internal type registries
are not shared, so ``py::cast<PyRblnSyncRuntime*>(handle)`` fails cross-DSO.
We bypass the type caster entirely by reading the C++ pointer directly from
pybind's instance layout: for a simple-layout pybind instance, the first
``void*`` after ``PyObject_HEAD`` is the C++ instance pointer. On CPython 3.12
x86_64 that's offset 16.
"""

from __future__ import annotations

import ctypes
import sys

import torch

from torch_rbln._internal.register_ops import (
    _add_op_module,
    _div_op_module,
    _mul_op_module,
    _neg_op_module,
    _sub_op_module,
)

_PYOBJECT_HEAD_SIZE = 16  # ob_refcnt (8) + ob_type (8) on 64-bit CPython.


def _raw_cpp_ptr(pybound_instance) -> int:
    """Return the raw C++ pointer held by a pybind11 instance as uintptr_t.

    Relies on pybind11's "simple layout" for single-inheritance types with a
    standard holder (``std::unique_ptr`` / ``std::shared_ptr``). The first
    pointer-sized slot after the Python object header is the C++ instance
    pointer. Verified against rebel's ``PyRblnSyncRuntime`` (pybind11 2.x).
    """
    addr = id(pybound_instance)
    raw = ctypes.c_void_p.from_address(addr + _PYOBJECT_HEAD_SIZE).value
    if raw is None:
        raise RuntimeError("unexpected null C++ pointer from pybind instance")
    return raw


def _drive_compile(op_module: torch.nn.Module, *sample_args: torch.Tensor):
    """Compile ``op_module`` with backend=rbln and return the DynamoRuntime.

    Uses the backend's ``_runtime_holder`` side-channel to recover the
    DynamoRuntime instance that Dynamo would otherwise hide inside its
    code cache.
    """
    holder: list = []
    options = {
        "tensor_parallel_size": 1,
        "disable_logger": True,
        "_runtime_holder": holder,
    }
    compiled = torch.compile(op_module, backend="rbln", dynamic=False, options=options)
    _ = compiled(*sample_args)
    if not holder:
        raise RuntimeError(
            "rbln_backend did not populate _runtime_holder; cannot bootstrap "
            f"C-kernel cache for {op_module.__class__.__name__}"
        )
    return holder[-1]


def _runtime_info(dyn_runtime) -> dict:
    """Pack everything the C++ miss path needs into a plain dict.

    Keeps ``dyn_runtime`` as a strong ref so the C++ side can stash it and
    keep the underlying rebel runtime alive.
    """
    executor = dyn_runtime._executor
    num_outputs = int(dyn_runtime._num_outputs)
    out_profiles = []
    for i in range(num_outputs):
        p = executor.get_output_profile(i)
        out_profiles.append(
            {
                "shape": list(p.shape),
                "dtype": p.dtype,  # e.g. "float16"
                "device": p.device,  # e.g. "rbln"
            }
        )
    return {
        "dyn_runtime": dyn_runtime,  # strong ref
        "runtime_raw_ptr": _raw_cpp_ptr(dyn_runtime._runtime_handle),
        "num_inputs": int(dyn_runtime._num_inputs),
        "num_outputs": num_outputs,
        "out_profiles": out_profiles,
    }


def build_add_runtime(sample_a: torch.Tensor, sample_b: torch.Tensor) -> dict:
    return _runtime_info(_drive_compile(_add_op_module, sample_a, sample_b))


def build_sub_runtime(sample_a: torch.Tensor, sample_b: torch.Tensor) -> dict:
    return _runtime_info(_drive_compile(_sub_op_module, sample_a, sample_b))


def build_mul_runtime(sample_a: torch.Tensor, sample_b: torch.Tensor) -> dict:
    return _runtime_info(_drive_compile(_mul_op_module, sample_a, sample_b))


def build_div_runtime(sample_a: torch.Tensor, sample_b: torch.Tensor) -> dict:
    return _runtime_info(_drive_compile(_div_op_module, sample_a, sample_b))


def build_neg_runtime(sample_a: torch.Tensor) -> dict:
    return _runtime_info(_drive_compile(_neg_op_module, sample_a))
