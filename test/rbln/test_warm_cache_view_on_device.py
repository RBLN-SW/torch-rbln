# Owner(s): ["module: PrivateUse1"]

"""
Regression coverage for the warm-cache × view-on-device interaction.

PR #47 (``feat(view-on-device)``) routes view recipes — narrow / permute /
expand / select / composite — through the device path by replacing the
view tensor with its base in the Python wrapper and emitting the recipe
inside the compiled FX graph. The runtime is therefore parameterized by
the *base* tensor's layout, not the view's.

Two interaction risks with the warm cache landed on PR #46:

  1. **Wrong-hit on view input**: the C++ shim's hit path passes the
     stack tensor's ``data_ptr()`` — i.e. the view's ``base + offset``
     pointer — to a runtime that, after view-on-device, expects ``base``.
     If the cache key carried only shape + dtype, a same-shape view of a
     different base (or a different recipe offset) would hit the entry
     and the runtime would double-apply the recipe to a wrong base.

  2. **Install poisoning on raw non-contig input**: even though
     ``compile_and_run_view_aware`` skips ``_install_warm_cache_pending``
     when ``has_views`` is True, the C++ side gets the pending key built
     from the raw stack tensor (a view). A future code path that
     bypasses the Python ``has_views`` gate must NOT land a non-contig
     key in the cache.

The fix is two-part:

  - ``build_cache_key`` writes input strides + storage_offset into the
    lookup key.
  - ``install_warmcache_from_pending`` rejects pending keys whose layout
    isn't contig + offset=0 (defense-in-depth against future install
    callers).

The tests below exercise the head_dim≥128 case that bypasses PR #47's
``align-fallback`` (which only kicks in for last-dim % 64 != 0), the
storage-alias case mentioned in ``prepare_args_view_aware``, and the
cache-size invariant for view-recipe calls.
"""

import pytest
import torch
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln  # noqa: F401 (registers the rbln backend)
from torch_rbln import _C


def _assert_fp16_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """rebel fp16 + view-on-device's ``contrib_dummy_cast`` round-trip
    introduces ~1 ULP drift per stage. Tolerance here is wide enough to
    absorb that drift while still catching the orders-of-magnitude error
    that a wrong-storage hit would produce."""
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.02)


@pytest.mark.test_set_ci
class TestRotateHalfHeadDim128(TestCase):
    """``rotate_half(x) = cat([-x[..., D//2:], x[..., :D//2]], dim=-1)``.

    For ``head_dim=128`` the halves are 64 elements each (multiples of 64),
    so the align-fallback added in PR #47 does NOT route this through
    cpu_fallback. View-on-device DOES kick in for the narrow views, so
    the second half (``storage_offset > 0``) is a textbook trigger for
    the wrong-hit scenario this fix targets.

    The test simulates the rotate_half compute pattern explicitly to
    avoid pulling in transformers' rotary helper while still exercising
    the same op sequence (neg + cat over narrow halves).
    """

    def setUp(self) -> None:
        _C._warmcache_clear()

    def _half_neg(self, head_dim: int) -> None:
        B, S = 2, 4
        x = torch.randn(B, S, head_dim, dtype=torch.float16, device="rbln")
        first = x[..., : head_dim // 2]  # storage_offset == 0
        second = x[..., head_dim // 2 :]  # storage_offset > 0
        # neg of each half — the rotate_half decomposition does this.
        out_first = -first
        out_second = -second
        ref_first = -x.cpu()[..., : head_dim // 2]
        ref_second = -x.cpu()[..., head_dim // 2 :]
        _assert_fp16_close(out_first.cpu(), ref_first)
        _assert_fp16_close(out_second.cpu(), ref_second)

    def test_head_dim_128(self) -> None:
        self._half_neg(128)

    def test_head_dim_256(self) -> None:
        self._half_neg(256)

    def test_head_dim_64_baseline_via_align_fallback(self) -> None:
        # head_dim=64 → halves are 32 (not multiple of 64) → align-fallback
        # kicks in and routes through cpu_fallback_path. Included as a
        # control case: this should already be correct on the pre-fix branch
        # because the warm-cache hit path isn't entered.
        self._half_neg(64)


@pytest.mark.test_set_ci
class TestStorageAliasInputs(TestCase):
    """``prepare_args_view_aware`` forces ``.contiguous()`` on every operand
    when two tensor args alias the same storage (rebel-compiler can't
    compile graphs with aliased inputs). The warm-cache install must not
    land an entry that would later serve a non-aliased call from the
    same shape with a stale aliased-input runtime — and vice versa.

    Aliased inputs always end up as ``.contiguous()`` copies in the
    Python wrapper, which yields contig+offset=0 key entries; the
    sanity check should accept those.
    """

    def setUp(self) -> None:
        _C._warmcache_clear()

    def test_alias_self_transpose_matmul(self) -> None:
        # ``mm(a, a.T)`` is the canonical alias case. ``a.T`` is non-contig,
        # so the C++ shim's pre-check on its own wouldn't shortcut. The
        # Python wrapper materializes both via ``.contiguous()`` before
        # compile.
        a = torch.randn(8, 16, dtype=torch.float16, device="rbln")
        out = torch.mm(a, a.t())
        ref = a.cpu() @ a.cpu().t()
        _assert_fp16_close(out.cpu(), ref)


@pytest.mark.test_set_ci
class TestWarmCacheSizeViewInteraction(TestCase):
    """``has_views=True`` calls must not install. ``has_views=False`` calls
    install once per (op, contig-shape) profile and reuse on subsequent hits.
    """

    def setUp(self) -> None:
        _C._warmcache_clear()

    def test_view_recipe_call_does_not_install(self) -> None:
        # Use a shape where view-on-device DOES kick in:
        # head_dim=128 narrow with offset>0.
        head_dim = 128
        x = torch.randn(2, 4, head_dim, dtype=torch.float16, device="rbln")
        baseline = _C._warmcache_size()
        # narrow with offset>0 → has_views=True path
        v = x[..., head_dim // 2 :]
        _ = -v
        after_view = _C._warmcache_size()
        self.assertEqual(
            after_view,
            baseline,
            f"view-recipe call should not install warm-cache; size {baseline} → {after_view}",
        )

    def test_contig_then_view_then_contig_is_correct(self) -> None:
        # Interleave contig and view-input calls with the same shape;
        # both must produce correct results regardless of order.
        head_dim = 128
        half = head_dim // 2
        x = torch.randn(2, 4, head_dim, dtype=torch.float16, device="rbln")

        # Contig with the half-shape on a fresh allocation.
        contig_a = torch.randn(2, 4, half, dtype=torch.float16, device="rbln")
        contig_b = torch.randn(2, 4, half, dtype=torch.float16, device="rbln")
        c_out = contig_a + contig_b
        c_ref = contig_a.cpu() + contig_b.cpu()
        _assert_fp16_close(c_out.cpu(), c_ref)

        # Now a view of the same shape — offset=0 (first half) and
        # offset>0 (second half). Both must be correct.
        first = x[..., :half]
        second = x[..., half:]
        first_out = first + first
        second_out = second + second
        _assert_fp16_close(first_out.cpu(), 2.0 * x.cpu()[..., :half])
        _assert_fp16_close(second_out.cpu(), 2.0 * x.cpu()[..., half:])

        # And a fresh contig call again — should reuse the contig entry.
        contig_c = torch.randn(2, 4, half, dtype=torch.float16, device="rbln")
        contig_d = torch.randn(2, 4, half, dtype=torch.float16, device="rbln")
        final_out = contig_c + contig_d
        final_ref = contig_c.cpu() + contig_d.cpu()
        _assert_fp16_close(final_out.cpu(), final_ref)


if __name__ == "__main__":
    run_tests()
