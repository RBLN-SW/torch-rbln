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
import torch_rbln._internal.monkey_patches as mp


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


class _Item:
    def __init__(self, nodeid, single_worker):
        self.nodeid = nodeid
        self.name = nodeid.rsplit("::", 1)[-1]
        self._single = single_worker

    def get_closest_marker(self, name):
        return object() if name == "single_worker" and self._single else None

    def add_marker(self, marker):  # _REBEL_XFAILS is empty, so never called
        raise AssertionError("unexpected add_marker")


class _Config:
    """Stub pytest config. Set ``workercount`` to simulate collection inside an xdist worker
    (the only place collection runs for a real parallel run); leave it None to simulate the
    controller / no-xdist / --collect-only, which execute nothing in parallel."""

    def __init__(self, *, workercount=None, numprocesses=0):
        self._numprocesses = numprocesses
        if workercount is not None:
            self.workerinput = {"workercount": workercount}

    def getoption(self, name, default=None):
        return self._numprocesses if name == "numprocesses" else default


@pytest.mark.test_set_ci
def test_single_worker_guard_fires_only_under_parallel():
    """The single_worker collection guard must fire only inside an xdist worker with
    workercount > 1 -- the only place tests actually execute in parallel. It must NOT fire on
    the controller / no-xdist / ``--collect-only -nN`` (which execute nothing), nor for a
    single worker (``-n1``, serial-safe)."""
    modify = _load_root_conftest().pytest_collection_modifyitems

    def single():
        return [_Item("f.py::test_a", single_worker=True)]

    # Real parallel run: an xdist worker with workercount > 1 collecting a single_worker
    # test -> loud error. THIS is the path that actually fires under `pytest -n32`.
    with pytest.raises(pytest.UsageError, match="single_worker"):
        modify(_Config(workercount=32), single())
    # -n1: a single worker is serial-safe (the run_tests.py single_worker pass) -> allowed.
    modify(_Config(workercount=1), single())
    # No worker context (controller / no xdist) -> allowed even with a single_worker test.
    modify(_Config(numprocesses=0), single())
    # --collect-only -nN: a controller-only dry run (numprocesses set, no workerinput) that
    # executes nothing -> must NOT false-positive.
    modify(_Config(numprocesses=8), single())
    # Parallel worker with single_worker already filtered out (run_tests.py parallel pass) -> ok.
    modify(_Config(workercount=32), [_Item("f.py::test_b", single_worker=False)])


@pytest.mark.test_set_ci
def test_torch_compile_patches_reapplied_by_fixture_teardown():
    """The keep_torch_compile_patches fixture must re-apply the import-time torch.compile
    patches in teardown when a test removed OR silently rebound them -- detected by callable
    identity (patches_active), not just the bookkeeping flags. Exercised through the fixture."""
    conftest = _load_root_conftest()
    try:
        # (a) hard removal -> patches_active() False -> teardown restores.
        mp.remove_all_patches()
        assert not mp.patches_active()
        _run_setup_teardown(conftest.keep_torch_compile_patches)
        assert mp.patches_active()
        # (b) silent rebind: flag still reads "patched", but torch.compile is no longer the
        # RBLN wrapper. The identity check must catch it and the fixture must restore it.
        torch.compile = lambda *a, **k: None
        assert mp._torch_compile_patched  # flag disagrees with reality...
        assert not mp.patches_active()  # ...identity check does not
        _run_setup_teardown(conftest.keep_torch_compile_patches)
        assert mp.patches_active()
    finally:
        if not mp.patches_active():
            mp.remove_all_patches()
            mp.apply_all_patches()


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


class _FakeReq:
    class _Node:
        @staticmethod
        def get_closest_marker(name):
            return None

    node = _Node()


@pytest.mark.test_set_ci
@pytest.mark.no_caching_allocator_reset  # the real autouse fixture must not call the patched fns
def test_caching_allocator_teardown_fails_loud_and_skips_flush_on_sync_error(monkeypatch):
    """A drain (synchronize) failure means the device may still have in-flight DMA, so the
    teardown must fail loud AND NOT flush -- freeing a faulted device's blocks would re-open
    the async-buffer race (and empty_cache clears the process-global WarmCache)."""
    conftest = _load_root_conftest()
    flushed = []
    monkeypatch.setattr(torch_rbln.device, "device_count", lambda: 1)

    def _sync_boom(idx):
        raise RuntimeError("sync fault")

    monkeypatch.setattr(torch.rbln, "synchronize", _sync_boom)
    monkeypatch.setattr(torch_rbln.memory, "empty_cache", lambda device: flushed.append(device))

    gen = conftest.reset_caching_allocator.__wrapped__(_FakeReq())
    next(gen)  # setup: yields
    with pytest.raises(RuntimeError, match="sync fault"):
        next(gen)  # teardown: drain fails -> raise, flush skipped
    assert flushed == []  # flush must be skipped when the drain failed


@pytest.mark.test_set_ci
@pytest.mark.no_caching_allocator_reset
def test_caching_allocator_teardown_reports_flush_error(monkeypatch):
    """When the drain succeeds, the flush runs; an empty_cache failure must be surfaced
    (aggregated), not hidden."""
    conftest = _load_root_conftest()
    monkeypatch.setattr(torch_rbln.device, "device_count", lambda: 1)
    monkeypatch.setattr(torch.rbln, "synchronize", lambda idx: None)

    def _flush_boom(device):
        raise RuntimeError("flush fault")

    monkeypatch.setattr(torch_rbln.memory, "empty_cache", _flush_boom)

    gen = conftest.reset_caching_allocator.__wrapped__(_FakeReq())
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
def test_deploy_and_nan_inf_gates_read_env_live():
    """The C++ deploy / nan_inf-disable gates must read the environment live (no process
    cache), so a set -> unset -> set toggle is observed every time. A static cache would
    latch the first value into the whole xdist worker and leak across tests."""
    deploy = torch_rbln._C._is_deploy_mode
    nan_inf = torch_rbln._C._is_nan_inf_check_disabled

    # deploy: ON -> unset -> ON must track live each time.
    assert _with_env("TORCH_RBLN_DEPLOY", "ON", deploy) is True
    assert _with_env("TORCH_RBLN_DEPLOY", None, deploy) is False
    assert _with_env("TORCH_RBLN_DEPLOY", "ON", deploy) is True

    # nan_inf disable: "nan_inf" -> unset -> "all" must track live each time.
    assert _with_env("TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK", "nan_inf", nan_inf) is True
    assert _with_env("TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK", None, nan_inf) is False
    assert _with_env("TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK", "all", nan_inf) is True


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
