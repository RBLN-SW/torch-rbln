# Owner(s): ["module: PrivateUse1"]

"""Unit tests for the CPU fast-path registry and the fp32 handlers under
``aten/src/ATen/native/rbln/fast_paths/``.

Coverage:
  1. Static-init registration — each handler installs itself at .so load
     time. Verified via ``torch_rbln._C._cpu_fast_path_registered``.
  2. Numerical correctness — fp32 ops on RBLN device hit ``cpu_fallback_rbln``
     (the dispatch shim's pre-check rejects non-fp16 dtypes), and the
     handler runs its plain host loop. Compare against the same op on a
     CPU reference tensor.
  3. Guard fallthrough — when a handler's preconditions (dtype / contig /
     shape / dim / exponent) are not met, it returns ``false`` and the
     generic ``redispatchBoxed(CPU)`` path runs instead. The result must
     still be numerically correct (i.e. the guard is purely a fast-path
     gate, never a correctness boundary).
"""

from __future__ import annotations

import pytest
import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln  # noqa: F401  (registers the rbln backend)
import torch_rbln._C as _C


@pytest.mark.test_set_ci
class TestCPUFastPathRegistration(TestCase):
    """Static-init registration check.

    Each ``fast_paths/*.cpp`` file ends with a
    ``REGISTER_RBLN_CPU_FAST_PATH(<op_name>, <handler>)`` macro that runs
    at .so load time. If the macro is dropped or the source file is
    excluded from the build, this test catches it before downstream
    suites silently fall through to the generic dispatcher.

    These assertions probe a process-wide pybind registry, not device-side
    tensor behaviour, so this class intentionally does NOT go through
    ``instantiate_device_type_tests`` — see ``test_warm_cache_internals.py``
    and ``test_file_offloading.py`` for the same precedent.
    """

    def test_rsqrt_out_registered(self):
        self.assertTrue(_C._cpu_fast_path_registered("aten::rsqrt.out"))

    def test_pow_tensor_scalar_out_registered(self):
        self.assertTrue(_C._cpu_fast_path_registered("aten::pow.Tensor_Scalar_out"))

    def test_mean_out_registered(self):
        self.assertTrue(_C._cpu_fast_path_registered("aten::mean.out"))

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


@pytest.mark.test_set_ci
class TestRsqrtFastPath(TestCase):
    """`aten::rsqrt.out` fast path — fp32 contig hot path + fp16 fall-through.

    Each fast-path handler in ``aten/src/ATen/native/rbln/fast_paths/`` is
    dtype-pinned (fp32 only); we test both the in-bounds dtype (handler
    fires) and the out-of-bounds dtype (boxed dispatcher takes over). The
    dtype is therefore intentionally hard-coded per method — using
    ``@dtypes(*SUPPORTED_DTYPES)`` would conflate the two contracts.
    """

    def test_fp32_contig_matches_cpu(self):
        # Fast path eligible: fp32 + contiguous + matching out shape.
        cpu_x = torch.tensor([1.0, 4.0, 9.0, 16.0, 25.0], dtype=torch.float32)
        rbln_x = cpu_x.to("rbln")
        rbln_out = torch.rsqrt(rbln_x)
        self.assertEqual(rbln_out.cpu(), torch.rsqrt(cpu_x))

    def test_fp16_falls_through_to_generic(self):
        # Handler guards on fp32; fp16 should fall through to the boxed
        # CPU dispatcher and still produce a correct rsqrt.
        cpu_x = torch.tensor([1.0, 4.0, 9.0, 16.0, 25.0], dtype=torch.float16)
        rbln_x = cpu_x.to("rbln")
        rbln_out = torch.rsqrt(rbln_x)
        self.assertEqual(rbln_out.cpu(), torch.rsqrt(cpu_x), rtol=0.005, atol=0.01)


@pytest.mark.test_set_ci
class TestPowSquaredFastPath(TestCase):
    """`aten::pow.Tensor_Scalar_out` fast path is fp32 + exp == 2 only; verify
    that both eligibility (fp32 contig, exp == 2) and fall-through (exp != 2)
    produce correct results."""

    def test_fp32_contig_exp_2_int_matches_cpu(self):
        # Fast path: fp32 contig + exp == 2 (int form).
        cpu_x = torch.tensor([1.0, 2.0, 3.0, -4.0], dtype=torch.float32)
        rbln_x = cpu_x.to("rbln")
        rbln_out = torch.pow(rbln_x, 2)
        self.assertEqual(rbln_out.cpu(), torch.pow(cpu_x, 2))

    def test_fp32_contig_exp_2_float_matches_cpu(self):
        # Same as above but exp expressed as float 2.0; handler accepts both.
        cpu_x = torch.tensor([1.0, 2.0, 3.0, -4.0], dtype=torch.float32)
        rbln_x = cpu_x.to("rbln")
        rbln_out = torch.pow(rbln_x, 2.0)
        self.assertEqual(rbln_out.cpu(), torch.pow(cpu_x, 2.0))

    def test_fp32_exp_3_falls_through(self):
        # Handler guards on exp == 2; exp == 3 must take the generic path.
        cpu_x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        rbln_x = cpu_x.to("rbln")
        rbln_out = torch.pow(rbln_x, 3)
        self.assertEqual(rbln_out.cpu(), torch.pow(cpu_x, 3))


@pytest.mark.test_set_ci
class TestMeanLastDimFastPath(TestCase):
    """`aten::mean.out` last-dim + keepdim fast path; verify eligibility
    (last dim, keepdim=True) vs fall-through (non-last dim, keepdim=False)."""

    def test_fp32_contig_lastdim_keepdim_matches_cpu(self):
        # Fast path: fp32 contig + dim=[-1] + keepdim=True + dtype=None.
        cpu_x = torch.randn(4, 8, dtype=torch.float32)
        rbln_x = cpu_x.to("rbln")
        rbln_out = rbln_x.mean(dim=-1, keepdim=True)
        self.assertEqual(rbln_out.cpu(), cpu_x.mean(dim=-1, keepdim=True), rtol=1e-5, atol=1e-6)

    def test_fp32_contig_lastdim_positive_index_matches_cpu(self):
        # Last dim expressed as positive index (== ndim - 1) is also accepted.
        cpu_x = torch.randn(3, 5, 7, dtype=torch.float32)
        rbln_x = cpu_x.to("rbln")
        rbln_out = rbln_x.mean(dim=2, keepdim=True)
        self.assertEqual(rbln_out.cpu(), cpu_x.mean(dim=2, keepdim=True), rtol=1e-5, atol=1e-6)

    def test_fp32_non_lastdim_falls_through(self):
        # Handler guards on dim == self_dim - 1; dim==0 must take generic.
        cpu_x = torch.randn(4, 8, dtype=torch.float32)
        rbln_x = cpu_x.to("rbln")
        rbln_out = rbln_x.mean(dim=0, keepdim=True)
        self.assertEqual(rbln_out.cpu(), cpu_x.mean(dim=0, keepdim=True), rtol=1e-5, atol=1e-6)

    def test_fp32_no_keepdim_falls_through(self):
        # Handler requires keepdim=True (out shape == self minus last dim,
        # which would change numel and break the borrow size math).
        cpu_x = torch.randn(4, 8, dtype=torch.float32)
        rbln_x = cpu_x.to("rbln")
        rbln_out = rbln_x.mean(dim=-1, keepdim=False)
        self.assertEqual(rbln_out.cpu(), cpu_x.mean(dim=-1, keepdim=False), rtol=1e-5, atol=1e-6)


@pytest.mark.test_set_ci
class TestIntSubFastPath(TestCase):
    """`aten::sub.out` integer fast path. The fp16-only device falls int ops back
    to CPU; the handler runs a host loop (int64 accumulate, stored in out dtype).
    Covers same-shape, scalar/last-dim broadcast, all int widths, and the
    alpha != 1 guard fall-through."""

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
        # `cs - pidx * partition_len` shape: [num_reqs, num_partition] - [num_partition].
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


@pytest.mark.test_set_ci
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


@pytest.mark.test_set_ci
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


instantiate_device_type_tests(TestRsqrtFastPath, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestPowSquaredFastPath, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestMeanLastDimFastPath, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestIntSubFastPath, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestIntMulFastPath, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestIntClampFastPath, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
