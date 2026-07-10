# Owner(s): ["module: PrivateUse1"]

"""Regression tests for the test-isolation guards in ``test/conftest.py``.

These pin the guards that stop one test's process-scoped mutations from leaking into
later tests on the same pytest-xdist worker (the class of bug behind the intermittent
profiler / pinned-copy CI failures)."""

import importlib.util
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
    """Stub config exposing only ``numprocesses`` (the resolved xdist worker count)."""

    def __init__(self, numprocesses):
        self._numprocesses = numprocesses

    def getoption(self, name, default=None):
        return self._numprocesses if name == "numprocesses" else default


@pytest.mark.test_set_ci
def test_single_worker_guard_fires_only_under_parallel():
    """The single_worker collection guard must key on worker COUNT, not xdist's dist mode:
    ``-n1`` also flips xdist into dist=load but a single worker is serial-safe (it is how
    run_tests.py runs the single_worker pass), so only >1 worker may reject a single_worker
    test."""
    modify = _load_root_conftest().pytest_collection_modifyitems

    def single():
        return [_Item("f.py::test_a", single_worker=True)]

    # >1 worker (genuine parallel) collecting a single_worker test -> loud error.
    with pytest.raises(pytest.UsageError, match="single_worker"):
        modify(_Config(numprocesses=8), single())
    # -n1: dist=load but a single worker is serial-safe -> allowed (regression for the
    # broken serial pass).
    modify(_Config(numprocesses=1), single())
    # xdist off (0 / None) -> allowed.
    modify(_Config(numprocesses=0), single())
    modify(_Config(numprocesses=None), single())
    # Parallel run with single_worker already filtered out (run_tests.py parallel pass) -> ok.
    modify(_Config(numprocesses=8), [_Item("f.py::test_b", single_worker=False)])


@pytest.mark.test_set_ci
def test_torch_compile_patches_reapplied_by_fixture_teardown():
    """The keep_torch_compile_patches autouse fixture must re-apply the import-time
    torch.compile patches in teardown when a test removed them -- exercised through the
    fixture's own teardown, not by calling apply_all_patches() directly."""
    conftest = _load_root_conftest()
    try:
        mp.remove_all_patches()
        assert not mp._torch_compile_patched
        _run_setup_teardown(conftest.keep_torch_compile_patches)
        assert mp._torch_compile_patched
        assert mp._torch_dynamo_reset_patched
    finally:
        if not (mp._torch_compile_patched and mp._torch_dynamo_reset_patched):
            mp.apply_all_patches()


@pytest.mark.test_set_ci
def test_restore_current_device_does_not_leak_initialized():
    """restore_current_device must not flip ``_initialized`` on a test that never touched
    the device: it snapshots the flag at setup and restores it in teardown, and skips the
    low-level set_device when the current index is unchanged."""
    conftest = _load_root_conftest()
    saved = _dev._initialized
    try:
        _dev._initialized = False
        _run_setup_teardown(conftest.restore_current_device)
        assert _dev._initialized is False  # teardown must not have set it True
    finally:
        _dev._initialized = saved


@pytest.mark.test_set_ci
@pytest.mark.no_caching_allocator_reset  # the real autouse fixture must not call the patched sync
def test_caching_allocator_teardown_fails_loud_on_sync_error(monkeypatch):
    """A synchronize() failure in teardown (a real device fault the test left behind) must
    propagate, not be swallowed and misreported as an empty_cache failure in a later test."""
    conftest = _load_root_conftest()

    class _Node:
        @staticmethod
        def get_closest_marker(name):
            return None

    class _Req:
        node = _Node()

    monkeypatch.setattr(torch_rbln.device, "device_count", lambda: 1)

    def _boom(idx):
        raise RuntimeError("synthetic device fault")

    monkeypatch.setattr(torch.rbln, "synchronize", _boom)

    gen = conftest.reset_caching_allocator.__wrapped__(_Req())
    next(gen)  # setup: just yields
    with pytest.raises(RuntimeError, match="synthetic device fault"):
        next(gen)  # teardown: drain -> synchronize raises -> propagates loudly


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
