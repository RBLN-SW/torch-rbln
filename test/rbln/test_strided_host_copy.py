# Owner(s): ["module: PrivateUse1"]

"""The descriptor path in non-contiguous host<->device ``copy_``.

Above a slab-size threshold the slabs go to the runtime as descriptors, skipping
the host gather pass; below it staging still wins, and on element-sized slabs
descriptors are catastrophically slower — so the threshold is the point of the
change and is asserted here, not just the values.

Which path ran is observable through the profiler's host-bounce counters: staging
records a bounce (it fills a host buffer), the descriptor path records none.
"""

from __future__ import annotations

import pytest
import torch
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

from test.utils_v2v import arange as _arange, ENGINE_DTYPES, eq as _eq, to_dev as _to_dev


# Slabs comfortably above / below torch_rbln's kMinStridedHostCopyBytes (16 KiB).
_BIG_SLAB_ELEMS = 32 * 1024  # 64 KiB at float16
_SMALL_SLAB_ELEMS = 512  # 1 KiB at float16


def _bounces(report: dict) -> int:
    """Total host-bounce incidents recorded in an explain() dump."""
    return int(report["hidden_host_bounce"]["total_count"])


@pytest.mark.test_set_ci
@pytest.mark.usefixtures("enable_deploy_mode")
class TestStridedHostCopy(TestCase):
    """Non-contiguous cpu<->rbln ``copy_`` pairs."""

    # ---- correctness, independent of which path is taken ----

    @dtypes(*ENGINE_DTYPES)
    def test_non_contiguous_source(self, dtype):
        src_full = _arange((4, 2 * _BIG_SLAB_ELEMS), dtype)
        src = src_full[:, :_BIG_SLAB_ELEMS]
        dst = _to_dev(torch.zeros(4, _BIG_SLAB_ELEMS, dtype=dtype))
        dst.copy_(src)
        _eq(dst, src)

    @dtypes(*ENGINE_DTYPES)
    def test_non_contiguous_device_destination(self, dtype):
        # The destination view is written in place; the gaps keep their contents.
        storage = _to_dev(torch.zeros(4, 2 * _BIG_SLAB_ELEMS, dtype=dtype))
        dst = storage[:, :_BIG_SLAB_ELEMS]
        src = _arange((4, _BIG_SLAB_ELEMS), dtype) + 1
        dst.copy_(src)
        _eq(dst, src)
        _eq(storage[:, _BIG_SLAB_ELEMS:], torch.zeros(4, _BIG_SLAB_ELEMS, dtype=dtype))

    @dtypes(*ENGINE_DTYPES)
    def test_non_contiguous_host_destination(self, dtype):
        storage = torch.zeros(4, 2 * _BIG_SLAB_ELEMS, dtype=dtype)
        dst = storage[:, :_BIG_SLAB_ELEMS]
        src = _to_dev(_arange((4, _BIG_SLAB_ELEMS), dtype) + 1)
        dst.copy_(src)
        self.assertTrue(torch.equal(dst, src.cpu()))
        self.assertTrue(torch.equal(storage[:, _BIG_SLAB_ELEMS:], torch.zeros(4, _BIG_SLAB_ELEMS, dtype=dtype)))

    def test_transposed_source(self):
        src = _arange((256, 256), torch.float32).t()
        dst = _to_dev(torch.zeros(256, 256, dtype=torch.float32))
        dst.copy_(src)
        _eq(dst, src.contiguous())

    def test_broadcast_source_is_expanded_not_repeated(self):
        src = _arange((4, 1), torch.float32).expand(4, _BIG_SLAB_ELEMS)
        dst = _to_dev(torch.zeros(4, _BIG_SLAB_ELEMS, dtype=torch.float32))
        dst.copy_(src)
        _eq(dst, src.contiguous())

    def test_storage_offset_is_honoured(self):
        src_full = _arange((4, 2 * _BIG_SLAB_ELEMS), torch.float32)
        src = src_full[:, _BIG_SLAB_ELEMS:]
        self.assertNotEqual(src.storage_offset(), 0)
        dst = _to_dev(torch.zeros(4, _BIG_SLAB_ELEMS, dtype=torch.float32))
        dst.copy_(src)
        _eq(dst, src)

    def test_dtype_cast_still_converts(self):
        # The entrypoints move bytes, so a cast has to keep taking the host path.
        src = _arange((4, _BIG_SLAB_ELEMS), torch.float32)[:, : _BIG_SLAB_ELEMS // 2]
        dst = _to_dev(torch.zeros(4, _BIG_SLAB_ELEMS // 2, dtype=torch.float16))
        dst.copy_(src)
        _eq(dst, src.to(torch.float16))

    # ---- the threshold ----

    def test_large_slabs_skip_the_host_staging_pass(self):
        src = _arange((4, 2 * _BIG_SLAB_ELEMS), torch.float16)[:, :_BIG_SLAB_ELEMS]
        dst = _to_dev(torch.zeros(4, _BIG_SLAB_ELEMS, dtype=torch.float16))
        with torch.rbln.explain() as p:
            dst.copy_(src)
        self.assertEqual(_bounces(p.dump()), 0, "slabs above the threshold must not stage")
        _eq(dst, src)

    def test_small_slabs_keep_staging(self):
        # Descriptors lose here: the fixed cost per slab exceeds the host pass.
        src = _arange((256, 2 * _SMALL_SLAB_ELEMS), torch.float16)[:, :_SMALL_SLAB_ELEMS]
        dst = _to_dev(torch.zeros(256, _SMALL_SLAB_ELEMS, dtype=torch.float16))
        with torch.rbln.explain() as p:
            dst.copy_(src)
        self.assertGreater(_bounces(p.dump()), 0, "slabs below the threshold must stage")
        _eq(dst, src)

    def test_element_sized_slabs_keep_staging(self):
        # The pathological shape: one descriptor per element is ~130x slower.
        src = _arange((512, 512), torch.float32).t()
        dst = _to_dev(torch.zeros(512, 512, dtype=torch.float32))
        with torch.rbln.explain() as p:
            dst.copy_(src)
        self.assertGreater(_bounces(p.dump()), 0, "element-sized slabs must stage")
        _eq(dst, src.contiguous())

    def test_device_to_host_large_slabs_skip_staging(self):
        src = _to_dev(_arange((4, _BIG_SLAB_ELEMS), torch.float16))
        storage = torch.zeros(4, 2 * _BIG_SLAB_ELEMS, dtype=torch.float16)
        dst = storage[:, :_BIG_SLAB_ELEMS]
        with torch.rbln.explain() as p:
            dst.copy_(src)
        self.assertEqual(_bounces(p.dump()), 0, "slabs above the threshold must not stage")
        self.assertTrue(torch.equal(dst, src.cpu()))


instantiate_device_type_tests(TestStridedHostCopy, globals(), only_for="privateuse1")

if __name__ == "__main__":
    run_tests()
