# Owner(s): ["module: PrivateUse1"]

"""Tests for the native ``aten::_foreach_copy_`` RBLN kernel.

``_foreach_copy_(self[], src[], non_blocking)`` copies ``src[i]`` into
``self[i]`` for every ``i``. The RBLN kernel batches every conversion-free,
same-device, same-shape pair into one ``rbln_memcpy_v2v_multi`` submit (a
scatter into N tensors collapses from N submits to 1) and falls back to a
per-pair ``copy_`` for broadcast / dtype-cast / cross-device pairs.

Batching submits the eligible copies together (no inter-copy ordering) and
after the inline fallback copies, so it must NOT change the observable result
versus strict list-order ``self[i].copy_(src[i])``. Coverage:

  - Basic correctness: N disjoint same-device pairs (the scatter fast path),
    across dtypes and over many pairs.
  - Mixed eligible / ineligible pairs (RBLN-to-RBLN plus CPU-to-RBLN).
  - Non-contiguous source / destination views.
  - Zero-numel pairs (skipped) and empty lists.
  - Disjoint views of one shared storage (e.g. kv_cache[0] / kv_cache[1]) —
    must stay correct without being forced off the fast path.
  - Alias / order-sensitive regressions where a tensor appears in both ``self``
    and ``src``, or destinations overlap (RAW/WAR and WAW). These force the
    sequential path; the result must still match list-order semantics.

Each test computes the reference by running the same op (with the same aliasing
wiring) on CPU tensors, which gives the list-order semantics, then compares the
device result bitwise.
"""

from __future__ import annotations

import pytest
import torch
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

from test.utils_v2v import arange as _arange, ENGINE_DTYPES, eq as _eq, to_dev as _to_dev


@pytest.mark.test_set_ci
@pytest.mark.usefixtures("enable_deploy_mode")
class TestForeachCopyV2V(TestCase):
    """Tests for the native ``aten::_foreach_copy_`` kernel reached through
    ``torch._foreach_copy_``."""

    # ---- basic correctness (the scatter fast path) ----

    @dtypes(*ENGINE_DTYPES)
    def test_basic_disjoint_pairs(self, dtype):
        """N independent same-device pairs — the common batched scatter."""
        n_pairs = 6
        dst_cpu = [_arange((4, 3), dtype) for _ in range(n_pairs)]
        src_cpu = [_arange((4, 3), dtype) + (i + 1) * 100 for i in range(n_pairs)]

        dst_dev = [_to_dev(d) for d in dst_cpu]
        torch._foreach_copy_(dst_dev, [_to_dev(s) for s in src_cpu])

        for got, want in zip(dst_dev, src_cpu):
            _eq(got, want)

    def test_many_pairs(self):
        """Many pairs in one call — exercises a wide V2VBatch."""
        n_pairs = 64
        src_cpu = [_arange((2, 8), torch.float32) + i for i in range(n_pairs)]
        dst_dev = [_to_dev(torch.zeros(2, 8, dtype=torch.float32)) for _ in range(n_pairs)]
        torch._foreach_copy_(dst_dev, [_to_dev(s) for s in src_cpu])
        for got, want in zip(dst_dev, src_cpu):
            _eq(got, want)

    def test_single_pair(self):
        src_cpu = _arange((5, 4), torch.float16) + 7
        dst_dev = [_to_dev(torch.zeros(5, 4, dtype=torch.float16))]
        torch._foreach_copy_(dst_dev, [_to_dev(src_cpu)])
        _eq(dst_dev[0], src_cpu)

    # ---- mixed eligible / ineligible pairs ----

    def test_mixed_rbln_and_cpu_source(self):
        """Some pairs are RBLN-to-RBLN (batched), some are CPU-to-RBLN
        (fallback). All disjoint, so order does not matter — result must match
        regardless of which path each pair takes."""
        a_src = _arange((4, 3), torch.float32) + 10
        b_src = _arange((4, 3), torch.float32) + 20  # delivered from CPU
        c_src = _arange((4, 3), torch.float32) + 30

        a = _to_dev(torch.zeros(4, 3, dtype=torch.float32))
        b = _to_dev(torch.zeros(4, 3, dtype=torch.float32))
        c = _to_dev(torch.zeros(4, 3, dtype=torch.float32))

        # b receives a CPU source -> fallback path; a, c are RBLN -> batched.
        torch._foreach_copy_([a, b, c], [_to_dev(a_src), b_src, _to_dev(c_src)])

        _eq(a, a_src)
        _eq(b, b_src)
        _eq(c, c_src)

    def test_mixed_dtype_cast_fallback(self):
        """A dtype-mismatched pair takes the per-pair copy_ (cast) fallback while
        the matching pair is batched."""
        same = _arange((3, 3), torch.float32) + 1
        cast_src = (_arange((3, 3), torch.float32) + 2).to(torch.float16)

        a = _to_dev(torch.zeros(3, 3, dtype=torch.float32))
        b = _to_dev(torch.zeros(3, 3, dtype=torch.float32))  # dst f32, src f16 -> cast
        torch._foreach_copy_([a, b], [_to_dev(same), _to_dev(cast_src)])

        _eq(a, same)
        _eq(b, cast_src.to(torch.float32))

    # ---- non-contiguous views ----

    def test_non_contig_dst(self):
        big = _to_dev(torch.zeros(4, 12, dtype=torch.float16))
        dst_view = big[:, :8]
        assert not dst_view.is_contiguous()
        src_cpu = _arange((4, 8), torch.float16) + 100
        torch._foreach_copy_([dst_view], [_to_dev(src_cpu)])
        _eq(dst_view, src_cpu)

    def test_non_contig_src(self):
        big_src_cpu = _arange((4, 16), torch.float32) + 5
        src_view = _to_dev(big_src_cpu)[:, :8]
        assert not src_view.is_contiguous()
        dst = _to_dev(torch.zeros(4, 8, dtype=torch.float32))
        torch._foreach_copy_([dst], [src_view])
        _eq(dst, big_src_cpu[:, :8])

    def test_both_non_contig_multi(self):
        big_dst = _to_dev(torch.zeros(6, 12, dtype=torch.int32))
        big_src_cpu = _arange((6, 16), torch.int32) + 1000
        dst_views = [big_dst[i : i + 2, :8] for i in (0, 2, 4)]
        src_views = [_to_dev(big_src_cpu)[i : i + 2, :8] for i in (0, 2, 4)]
        torch._foreach_copy_(dst_views, src_views)
        for v, i in zip(dst_views, (0, 2, 4)):
            _eq(v, big_src_cpu[i : i + 2, :8])

    # ---- zero-numel / empty ----

    def test_zero_numel_pair_skipped(self):
        """A 0-numel pair is a no-op and must not disturb the other pairs."""
        empty_dst = _to_dev(torch.empty(0, 4, dtype=torch.float32))
        empty_src = _to_dev(torch.empty(0, 4, dtype=torch.float32))
        real_src = _arange((3, 4), torch.float32) + 9
        real_dst = _to_dev(torch.zeros(3, 4, dtype=torch.float32))
        torch._foreach_copy_([empty_dst, real_dst], [empty_src, _to_dev(real_src)])
        _eq(real_dst, real_src)
        assert empty_dst.numel() == 0

    def test_empty_lists_rejected_upstream(self):
        """Empty tensor lists are rejected by PyTorch before reaching the RBLN
        kernel, so the op is never invoked with zero pairs."""
        with pytest.raises(RuntimeError, match="at least one tensor"):
            torch._foreach_copy_([], [])

    # ---- disjoint views of one storage must stay on the fast path ----

    def test_disjoint_views_same_storage(self):
        """Mirrors the KV-cache scatter: per-call destinations are disjoint
        slices of one shared buffer (kv[0] / kv[1]), sources are disjoint
        slices of one staging buffer. No real overlap, so the result must be
        correct (and the kernel must not treat shared-storage-but-disjoint as
        aliasing)."""
        n_layers = 3
        # One destination buffer shaped [2, L, 8]; pair k writes [kv, layer].
        dst_base = _to_dev(torch.zeros(2, n_layers, 8, dtype=torch.float32))
        # One staging buffer with matching layout, prefilled with a ramp.
        stage_cpu = _arange((2, n_layers, 8), torch.float32) + 1
        stage_dev = _to_dev(stage_cpu)

        dsts = []
        srcs = []
        for layer in range(n_layers):
            dsts.append(dst_base[0, layer])
            srcs.append(stage_dev[0, layer])
            dsts.append(dst_base[1, layer])
            srcs.append(stage_dev[1, layer])
        torch._foreach_copy_(dsts, srcs)
        _eq(dst_base, stage_cpu)

    # ---- alias / order-sensitive regressions ----

    def test_alias_src_is_other_pairs_dst(self):
        """Regression for the reported bug: a tensor is both an eligible copy's
        source and a fallback copy's destination.

        ``_foreach_copy_([a, b], [b, cpu_src])`` — list-order semantics give
        ``a == original b`` and ``b == cpu_src``. A naive batched path defers
        ``a <- b`` past the immediate ``b <- cpu_src``, so ``a`` would wrongly
        read the mutated ``b``."""
        a0 = _arange((2, 4), torch.float32) + 1
        b0 = _arange((2, 4), torch.float32) + 50
        cpu_src = _arange((2, 4), torch.float32) + 900

        # CPU reference with identical aliasing wiring.
        a_ref, b_ref = a0.clone(), b0.clone()
        torch._foreach_copy_([a_ref, b_ref], [b_ref, cpu_src])

        a = _to_dev(a0)
        b = _to_dev(b0)
        torch._foreach_copy_([a, b], [b, cpu_src])  # cpu_src forces b's fallback

        _eq(a, a_ref)  # must be original b, not cpu_src
        _eq(b, b_ref)

    def test_alias_swap_within_batch(self):
        """Both pairs are batch-eligible and alias each other:
        ``_foreach_copy_([a, b], [b, a])``. List order copies ``a <- b`` then
        ``b <- (new) a`` so both end as the original ``b``. The batched path has
        no inter-copy ordering, so this must drop to the sequential path."""
        a0 = _arange((3, 4), torch.float32) + 1
        b0 = _arange((3, 4), torch.float32) + 100

        a_ref, b_ref = a0.clone(), b0.clone()
        torch._foreach_copy_([a_ref, b_ref], [b_ref, a_ref])

        a = _to_dev(a0)
        b = _to_dev(b0)
        torch._foreach_copy_([a, b], [b, a])

        _eq(a, a_ref)
        _eq(b, b_ref)

    def test_overlapping_destinations_waw(self):
        """Two destinations are overlapping slices of one buffer fed by disjoint
        sources. List order makes the later write win on the overlap; the
        batched path must not reorder them."""
        base0 = _arange((8,), torch.float32)
        x = _arange((4,), torch.float32) + 100
        y = _arange((4,), torch.float32) + 200

        base_ref = base0.clone()
        torch._foreach_copy_([base_ref[0:4], base_ref[2:6]], [x, y])

        base_dev = _to_dev(base0)
        torch._foreach_copy_([base_dev[0:4], base_dev[2:6]], [_to_dev(x), _to_dev(y)])

        _eq(base_dev, base_ref)

    def test_identity_self_equals_src(self):
        """Within-pair identity (``self[i] is src[i]``) is a no-op and must be
        left unchanged — within-pair overlap is not a reorder hazard."""
        a0 = _arange((4, 4), torch.float32) + 3
        a = _to_dev(a0)
        torch._foreach_copy_([a], [a])
        _eq(a, a0)


instantiate_device_type_tests(TestForeachCopyV2V, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
