# Owner(s): ["module: PrivateUse1"]

"""Regression tests for the test-isolation guards in ``test/conftest.py``.

These pin the guards that stop one test's process-scoped mutations from leaking into
later tests on the same pytest-xdist worker (the class of bug behind the intermittent
profiler / pinned-copy CI failures)."""

import importlib.util
import os
import sys

import pytest
import torch

import torch_rbln


# The `torch_rbln.device` package re-exports a `device` class that shadows the `device`
# submodule; grab the module (which owns the `_initialized` global) via sys.modules.
_dev = sys.modules[torch_rbln.device.set_device.__module__]


def _load_root_conftest():
    spec = importlib.util.spec_from_file_location("rbln_root_conftest", "test/conftest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_setup_teardown(fixture, *args):
    """Drive an autouse generator fixture through setup + teardown so its teardown side
    effects can be asserted directly."""
    gen = fixture.__wrapped__(*args)
    next(gen)  # setup, up to yield
    try:
        next(gen)  # teardown, past yield
    except StopIteration:
        pass


@pytest.mark.test_set_ci
def test_restore_current_device_does_not_leak_initialized():
    """restore_current_device must not flip ``_initialized`` on a test that never touched the
    device: it snapshots the flag and restores it, and skips set_device when the index is
    unchanged."""
    conftest = _load_root_conftest()
    saved = _dev._initialized
    try:
        _dev._initialized = False
        _run_setup_teardown(conftest.restore_current_device)
        assert _dev._initialized is False  # teardown must not have set it True
    finally:
        _dev._initialized = saved


# The device functions are patched directly on the module (not via the monkeypatch fixture)
# and restored in a finally inside the test body, so they are back to real BEFORE the real
# autouse restore_current_device fixture tears down and would otherwise call the mock.
@pytest.mark.test_set_ci
def test_restore_current_device_restores_changed_index():
    """When a test moves the current device index, restore_current_device must set it back in
    teardown -- and only then (device-independent via patched device functions)."""
    conftest = _load_root_conftest()
    state = {"idx": 0}
    set_calls = []
    orig = (_dev.device_count, _dev.current_device, _dev.set_device)
    try:
        _dev.device_count = lambda: 2
        _dev.current_device = lambda: state["idx"]

        def _set(i):
            state["idx"] = i
            set_calls.append(i)

        _dev.set_device = _set
        gen = conftest.restore_current_device.__wrapped__()
        next(gen)  # setup: saved = current_device() = 0
        state["idx"] = 1  # a "test" moved the current device
        try:
            next(gen)  # teardown: current_device() 1 != 0 -> set_device(0)
        except StopIteration:
            pass
        assert set_calls == [0]
        assert state["idx"] == 0
    finally:
        _dev.device_count, _dev.current_device, _dev.set_device = orig


@pytest.mark.test_set_ci
def test_restore_current_device_propagates_restore_failure():
    """A device-index restore failure must propagate as a teardown error, not be swallowed
    while _initialized is rolled back -- which would hand later tests a mismatched
    (index moved, flag old) state."""
    conftest = _load_root_conftest()
    state = {"idx": 0}
    orig = (_dev.device_count, _dev.current_device, _dev.set_device)
    saved_flag = _dev._initialized
    try:
        _dev.device_count = lambda: 2
        _dev.current_device = lambda: state["idx"]

        def _boom(i):
            raise RuntimeError("synthetic set_device failure")

        _dev.set_device = _boom
        _dev._initialized = False  # pre-test flag
        gen = conftest.restore_current_device.__wrapped__()
        next(gen)  # setup snapshots saved_initialized = False
        state["idx"] = 1  # test moved the device...
        _dev._initialized = True  # ...and initialized it
        with pytest.raises(RuntimeError, match="synthetic set_device failure"):
            next(gen)  # teardown: set_device raises before the flag rollback
        assert _dev._initialized is True  # NOT rolled back to the pre-test False
    finally:
        _dev.device_count, _dev.current_device, _dev.set_device = orig
        _dev._initialized = saved_flag


@pytest.mark.test_set_ci
def test_caching_allocator_teardown_fails_loud_and_skips_flush_on_sync_error(monkeypatch):
    """A drain (synchronize) failure means the device may still have in-flight DMA, so the
    teardown must fail loud AND NOT flush -- freeing a faulted device's blocks would re-open
    the async-buffer race (and empty_cache clears the process-global WarmCache)."""
    conftest = _load_root_conftest()
    flushed = []

    def _sync_boom(idx):
        raise RuntimeError("sync fault")

    # context() restores the patches on block exit -- success OR assertion failure -- before
    # the real autouse reset_caching_allocator teardown runs, so it never calls the patched fns.
    with monkeypatch.context() as m:
        m.setattr(torch_rbln.device, "device_count", lambda: 1)
        m.setattr(torch.rbln, "synchronize", _sync_boom)
        m.setattr(torch_rbln.memory, "empty_cache", lambda device: flushed.append(device))

        gen = conftest.reset_caching_allocator.__wrapped__()
        next(gen)  # setup: yields
        with pytest.raises(RuntimeError, match="sync fault"):
            next(gen)  # teardown: drain fails -> raise, flush skipped
        assert flushed == []  # flush must be skipped when the drain failed


@pytest.mark.test_set_ci
def test_caching_allocator_teardown_reports_flush_error(monkeypatch):
    """When the drain succeeds, the flush runs; an empty_cache failure must be surfaced
    (aggregated), not hidden."""
    conftest = _load_root_conftest()

    def _flush_boom(device):
        raise RuntimeError("flush fault")

    # context() restores the patches on block exit before the real autouse teardown runs.
    with monkeypatch.context() as m:
        m.setattr(torch_rbln.device, "device_count", lambda: 1)
        m.setattr(torch.rbln, "synchronize", lambda idx: None)
        m.setattr(torch_rbln.memory, "empty_cache", _flush_boom)

        gen = conftest.reset_caching_allocator.__wrapped__()
        next(gen)  # setup: yields
        with pytest.raises(RuntimeError, match="flush fault"):
            next(gen)  # teardown: drain ok -> flush runs -> its failure surfaced


def _with_env(var, value, fn):
    """Run fn() with env var set to value (None = unset), then restore the prior value."""
    old = os.environ.get(var)
    try:
        if value is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = value
        return fn()
    finally:
        if old is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = old


@pytest.mark.test_set_ci
def test_python_disabled_fallback_gate_reads_env_live():
    """The Python cold-path gate must read the env live too, matching the C++ warm-path gate
    (else warm/cold dispatch disagree and the value latches across an xdist worker). It was
    previously @lru_cache'd."""
    from torch_rbln._internal.ops_utils import _parse_disabled_fallback_cases

    def has_nan_inf():
        return "nan_inf" in _parse_disabled_fallback_cases()

    # "nan_inf" -> unset -> "all" must each be observed live (no lru_cache latch).
    assert _with_env("TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK", "nan_inf", has_nan_inf) is True
    assert _with_env("TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK", None, has_nan_inf) is False
    assert _with_env("TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK", "all", has_nan_inf) is True


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
