# Owner(s): ["module: PrivateUse1"]

"""
Float32 ops run as the process's float32 precision says: on the device, which
computes floats in dlfloat16, by default, or on the CPU, as float32, once
``rbln.set_float32_precision("exact")`` asks for it.
"""

import pytest
import torch
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln  # noqa: F401 (registers the rbln backend)
from test.utils import run_in_isolated_process


def _float32_mul_worker(precision, on_device):
    import rbln
    from torch_rbln import _C

    rbln.set_float32_precision(precision)
    x, y = torch.randn(8, 64), torch.randn(8, 64)
    _C._dispatch_fallback_reasons_reset()
    got = (x.to("rbln:0") * y.to("rbln:0")).cpu()
    dtype_fallbacks = _C._dispatch_fallback_reasons()[0]
    if on_device:
        assert dtype_fallbacks == 0, dtype_fallbacks
        torch.testing.assert_close(got, x * y, rtol=1e-2, atol=1e-2)
        assert not torch.equal(got, x * y)
    else:
        assert dtype_fallbacks == 1, dtype_fallbacks
        assert torch.equal(got, x * y)


def _keyed_by_precision_worker():
    import rbln
    from torch_rbln import _C
    from torch_rbln._internal import compile_cache

    h = torch.randn(4, 64, dtype=torch.float16).to("rbln:0")
    torch.add(h, h)
    compiled, warm = len(compile_cache._compiled_op_cache), _C._warmcache_size()
    rbln.set_float32_precision("exact")
    torch.add(h, h)
    assert len(compile_cache._compiled_op_cache) == compiled + 1
    assert _C._warmcache_size() == warm + 1


@pytest.mark.test_set_ci
class TestFloat32Precision(TestCase):
    def test_float32_ops_run_on_the_device_by_default(self):
        run_in_isolated_process(_float32_mul_worker, "device", True)

    def test_exact_float32_ops_run_on_the_cpu(self):
        run_in_isolated_process(_float32_mul_worker, "exact", False)

    def test_an_op_compiled_at_one_precision_is_not_run_at_the_other(self):
        run_in_isolated_process(_keyed_by_precision_worker)


if __name__ == "__main__":
    run_tests()
