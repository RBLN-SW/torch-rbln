# Owner(s): ["module: PrivateUse1"]

"""Unit tests for the CPU fast-path registry and the integer handlers under
``aten/src/ATen/native/rbln/fast_paths/``.

Coverage:
  1. Static-init registration — each handler installs itself at .so load
     time. Verified via ``torch_rbln._C._cpu_fast_path_registered``.
  2. Numerical correctness — the handlers run inside ``cpu_fallback_rbln``,
     which integer ops reach on every call (int dtypes are outside the
     dispatch catalog). Compare against the same op on a CPU reference
     tensor.
  3. Guard fallthrough — when a handler's preconditions (dtype / contig /
     shape / broadcast / alpha) are not met, it returns ``false`` and the
     generic ``redispatchBoxed(CPU)`` path runs instead. The result must
     still be numerically correct (i.e. the guard is purely a fast-path
     gate, never a correctness boundary).
"""

from __future__ import annotations

import pytest
import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

from torch_rbln import _C


class TestCPUFastPathRegistration(TestCase):
    """Static-init registration check.

    Each ``fast_paths/*.cpp`` file ends with a
    ``REGISTER_RBLN_CPU_FAST_PATH(<op_name>, <handler>)`` macro that runs
    at .so load time. If the macro is dropped or the source file is
    excluded from the build, this test catches it before downstream
    suites silently fall through to the generic dispatcher.

    These assertions probe a process-wide pybind registry, not device-side
    tensor behavior, so this class intentionally does NOT go through
    ``instantiate_device_type_tests`` — see ``test_warm_cache_internals.py``
    and ``test_file_offloading.py`` for the same precedent.
    """

    def test_sub_out_registered(self):
        self.assertTrue(_C._cpu_fast_path_registered("aten::sub.out"))

    def test_mul_out_registered(self):
        self.assertTrue(_C._cpu_fast_path_registered("aten::mul.out"))

    def test_clamp_out_registered(self):
        self.assertTrue(_C._cpu_fast_path_registered("aten::clamp.out"))

    def test_unregistered_op_returns_false(self):
        # add.out goes through the dispatch shim, not the cpu_fallback fast-path
        # registry — should not be present here.
        self.assertFalse(_C._cpu_fast_path_registered("aten::add.out"))
        self.assertFalse(_C._cpu_fast_path_registered("aten::nonexistent.op"))


@pytest.mark.npu(count=1)
class TestIntSubFastPath(TestCase):
    """`aten::sub.out` integer fast path. Int dtypes are outside the dispatch catalog,
    so int ops fall back to CPU; the handler runs a host loop (int64 accumulate, stored
    in out dtype). Covers same-shape, scalar/last-dim broadcast, all int widths, and the
    alpha != 1 guard fall-through.
    """

    def test_int32_same_shape_matches_cpu(self):
        cpu_a = torch.tensor([5, 3, 9, 1], dtype=torch.int32)
        cpu_b = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        ra, rb = cpu_a.to("rbln"), cpu_b.to("rbln")
        self.assertEqual((ra - rb).cpu(), cpu_a - cpu_b)

    def test_int64_same_shape_matches_cpu(self):
        cpu_a = torch.tensor([5, 3, 9, 1], dtype=torch.int64)
        cpu_b = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        ra, rb = cpu_a.to("rbln"), cpu_b.to("rbln")
        self.assertEqual((ra - rb).cpu(), cpu_a - cpu_b)

    def test_int16_same_shape_matches_cpu(self):
        cpu_a = torch.tensor([5, 3, 9, 1], dtype=torch.int16)
        cpu_b = torch.tensor([1, 2, 3, 4], dtype=torch.int16)
        ra, rb = cpu_a.to("rbln"), cpu_b.to("rbln")
        self.assertEqual((ra - rb).cpu(), cpu_a - cpu_b)

    def test_int32_scalar_broadcast_matches_cpu(self):
        # `logits_indices = query_start_loc[1:] - 1` shape.
        cpu_a = torch.tensor([5, 3, 9], dtype=torch.int32)
        ra = cpu_a.to("rbln")
        self.assertEqual((ra - 1).cpu(), cpu_a - 1)

    def test_int32_lastdim_broadcast_matches_cpu(self):
        # `cs - pidx * partition_len` shape: [num_reqs, num_partition]
        # - [num_partition].
        cpu_a = torch.tensor([[10, 20, 30]], dtype=torch.int32)
        cpu_b = torch.tensor([1, 2, 3], dtype=torch.int32)
        ra, rb = cpu_a.to("rbln"), cpu_b.to("rbln")
        self.assertEqual((ra - rb).cpu(), cpu_a - cpu_b)

    def test_alpha_not_one_falls_through(self):
        # Handler guards on alpha == 1; alpha=2 must take the generic path.
        cpu_a = torch.tensor([5, 3, 9, 1], dtype=torch.int32)
        cpu_b = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        ra, rb = cpu_a.to("rbln"), cpu_b.to("rbln")
        self.assertEqual(torch.sub(ra, rb, alpha=2).cpu(), torch.sub(cpu_a, cpu_b, alpha=2))


@pytest.mark.npu(count=1)
class TestIntMulFastPath(TestCase):
    """`aten::mul.out` integer fast path; same-shape and scalar broadcast."""

    def test_int32_same_shape_matches_cpu(self):
        cpu_a = torch.tensor([5, 3, 9, 1], dtype=torch.int32)
        cpu_b = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        ra, rb = cpu_a.to("rbln"), cpu_b.to("rbln")
        self.assertEqual((ra * rb).cpu(), cpu_a * cpu_b)

    def test_int32_scalar_broadcast_matches_cpu(self):
        # `pidx * partition_len` shape.
        cpu_a = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        ra = cpu_a.to("rbln")
        self.assertEqual((ra * 1024).cpu(), cpu_a * 1024)


@pytest.mark.npu(count=1)
class TestIntClampFastPath(TestCase):
    """`aten::clamp.out` integer fast path; min+max, min-only, max-only."""

    def test_int32_min_max_matches_cpu(self):
        # `clamp(.., 0, partition_len)` shape.
        cpu_x = torch.tensor([-5, 0, 500, 2000], dtype=torch.int32)
        rx = cpu_x.to("rbln")
        self.assertEqual(torch.clamp(rx, 0, 1024).cpu(), torch.clamp(cpu_x, 0, 1024))

    def test_int32_min_only_matches_cpu(self):
        cpu_x = torch.tensor([-5, 0, 7], dtype=torch.int32)
        rx = cpu_x.to("rbln")
        self.assertEqual(torch.clamp(rx, min=0).cpu(), torch.clamp(cpu_x, min=0))

    def test_int32_max_only_matches_cpu(self):
        cpu_x = torch.tensor([-5, 0, 2000], dtype=torch.int32)
        rx = cpu_x.to("rbln")
        self.assertEqual(torch.clamp(rx, max=1024).cpu(), torch.clamp(cpu_x, max=1024))


instantiate_device_type_tests(TestIntSubFastPath, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestIntMulFastPath, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestIntClampFastPath, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
