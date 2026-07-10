# Owner(s): ["module: PrivateUse1"]

"""Regression tests for the test-isolation guards in ``test/conftest.py``.

These pin the guards that stop one test's process-scoped mutations from leaking into
later tests on the same pytest-xdist worker (the class of bug behind the intermittent
profiler / pinned-copy CI failures)."""

import importlib.util

import pytest

import torch_rbln._internal.monkey_patches as mp


def _load_root_conftest():
    spec = importlib.util.spec_from_file_location("rbln_root_conftest", "test/conftest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    def __init__(self, dist):
        self._dist = dist

    def getoption(self, name, default=None):
        return self._dist if name == "dist" else default


def test_single_worker_guard_fires_only_under_parallel():
    modify = _load_root_conftest().pytest_collection_modifyitems
    single = [_Item("f.py::test_a", single_worker=True)]
    plain = [_Item("f.py::test_b", single_worker=False)]
    # parallel run (dist != "no") that still collects a single_worker test -> loud error
    with pytest.raises(pytest.UsageError, match="single_worker"):
        modify(_Config("load"), single)
    # parallel run with single_worker already filtered (run_tests.py parallel pass) -> ok
    modify(_Config("load"), plain)
    # serial run (dist == "no") -> single_worker allowed
    modify(_Config("no"), single)


def test_torch_compile_patches_reapply_after_removal():
    # The keep_torch_compile_patches autouse guard relies on apply_all_patches() restoring
    # the import-time patched state after a test calls remove_all_patches().
    mp.remove_all_patches()
    assert not mp._torch_compile_patched
    mp.apply_all_patches()
    assert mp._torch_compile_patched
    assert mp._torch_dynamo_reset_patched


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
