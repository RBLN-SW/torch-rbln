# Owner(s): ["module: PrivateUse1"]

"""
The process keeps TORCH_RBLN_COMPILED_OPS compiled ops, letting go of the least
recently run with what the warm cache holds of them, so that a long run over
many argument profiles holds a bounded number of executors and programs.
"""

import os
from unittest import mock

import pytest
import torch
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln  # noqa: F401 (registers the rbln backend)
from test.utils import run_in_isolated_process


def _keeps_the_most_recently_run_ops_worker():
    from torch_rbln import _C
    from torch_rbln._internal import compile_cache

    compiled = []
    compile_op = compile_cache._compile

    def counting(module, args, kwargs, device_index):
        compiled.append(tuple(args[0].shape))
        return compile_op(module, args, kwargs, device_index)

    compile_cache._compile = counting
    for rows in (1, 2, 3, 1, 4, 1, 2):
        x = torch.randn(rows, 64, dtype=torch.float16)
        torch.testing.assert_close(torch.abs(x.to("rbln:0")).cpu(), torch.abs(x))
        assert len(compile_cache._compiled_op_cache) <= 3
        assert _C._warmcache_size() <= 3
    # The op of one row runs between the others, from the warm cache, so it stays.
    assert compiled == [(1, 64), (2, 64), (3, 64), (4, 64), (2, 64)], compiled


@pytest.mark.test_set_ci
class TestCompiledOpLimit(TestCase):
    def test_a_full_cache_lets_go_of_the_least_recently_run(self):
        with mock.patch.dict(os.environ, {"TORCH_RBLN_COMPILED_OPS": "3", "TORCH_RBLN_DEPLOY": "ON"}):
            run_in_isolated_process(_keeps_the_most_recently_run_ops_worker)


if __name__ == "__main__":
    run_tests()
