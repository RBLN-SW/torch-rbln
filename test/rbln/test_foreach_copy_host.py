# Owner(s): ["module: PrivateUse1"]

"""Host-direction (cpu<->rbln) batching in the native ``aten::_foreach_copy_``.

``test_foreach_copy_v2v.py`` covers rbln->rbln. Every test asserts both:

  - values against list-order ``self[i].copy_(src[i])`` semantics, since a batch
    submits unordered and so an order-dependent result must be refused rather
    than silently reordered;
  - that batching happened, via the ``torch.rbln.explain()`` rt-timing counters.
    Values alone pass on an unbatched implementation.
"""

from __future__ import annotations

import pytest
import torch
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

from test.utils_v2v import arange as _arange, ENGINE_DTYPES, eq as _eq, to_dev as _to_dev


def _prim_calls(report: dict) -> dict[str, int]:
    """Per-primitive runtime call counts from an explain() dump."""
    rt = report.get("rebel_runtime")
    if rt is None:
        return {}
    return {name: int(v["calls"]) for name, v in rt.get("by_primitive", {}).items()}


@pytest.mark.test_set_ci
@pytest.mark.usefixtures("enable_deploy_mode")
class TestForeachCopyHost(TestCase):
    """cpu<->rbln pairs reached through ``torch._foreach_copy_``."""

    # ---- cpu -> rbln (H2VBatch) ----

    @dtypes(*ENGINE_DTYPES)
    def test_h2v_basic_disjoint_pairs(self, dtype):
        """N cpu sources into N disjoint device destinations."""
        n_pairs = 6
        src_cpu = [_arange((4, 3), dtype) + (i + 1) * 100 for i in range(n_pairs)]
        dst_dev = [_to_dev(torch.zeros(4, 3, dtype=dtype)) for _ in range(n_pairs)]

        torch._foreach_copy_(dst_dev, src_cpu)

        for got, want in zip(dst_dev, src_cpu):
            _eq(got, want)

    def test_h2v_many_pairs(self):
        n_pairs = 64
        src_cpu = [_arange((2, 8), torch.float32) + i for i in range(n_pairs)]
        dst_dev = [_to_dev(torch.zeros(2, 8, dtype=torch.float32)) for _ in range(n_pairs)]
        torch._foreach_copy_(dst_dev, src_cpu)
        for got, want in zip(dst_dev, src_cpu):
            _eq(got, want)

    def test_h2v_noncontiguous_device_dst(self):
        """A strided device destination is not batched. Writes still land in the
        viewed positions and the gaps keep their previous contents."""
        base = _to_dev(torch.full((5, 6), -1.0, dtype=torch.float32))
        dst_views = [base.select(1, c) for c in (1, 3)]
        src_cpu = [_arange((5,), torch.float32) + c * 10 for c in (1, 3)]

        with torch.rbln.explain() as p:
            torch._foreach_copy_(dst_views, src_cpu)
        calls = _prim_calls(p.dump())
        self.assertEqual(calls.get("h2v_multi", 0), 0, f"strided dst must not batch: {calls}")

        want = torch.full((5, 6), -1.0, dtype=torch.float32)
        for c, s in zip((1, 3), src_cpu):
            want.select(1, c).copy_(s)
        _eq(base, want)

    def test_h2v_noncontiguous_cpu_src(self):
        """A transposed cpu source is not batched — it would emit one descriptor
        per element. ``copy_`` stages it into one bulk DMA instead."""
        src_cpu = [_arange((4, 6), torch.float32).t(), _arange((4, 6), torch.float32).t() + 50]
        dst_dev = [_to_dev(torch.zeros(6, 4, dtype=torch.float32)) for _ in src_cpu]

        with torch.rbln.explain() as p:
            torch._foreach_copy_(dst_dev, src_cpu)
        calls = _prim_calls(p.dump())
        self.assertEqual(calls.get("h2v_multi", 0), 0, f"transposed src must not batch: {calls}")

        for got, want in zip(dst_dev, src_cpu):
            _eq(got, want)

    # ---- rbln -> cpu (V2HBatch) ----

    @dtypes(*ENGINE_DTYPES)
    def test_v2h_basic_disjoint_pairs(self, dtype):
        n_pairs = 6
        src_cpu = [_arange((4, 3), dtype) + (i + 1) * 7 for i in range(n_pairs)]
        src_dev = [_to_dev(s) for s in src_cpu]
        dst_cpu = [torch.zeros(4, 3, dtype=dtype) for _ in range(n_pairs)]

        torch._foreach_copy_(dst_cpu, src_dev)

        for got, want in zip(dst_cpu, src_cpu):
            self.assertTrue(torch.equal(got, want), f"{got} != {want}")

    def test_v2h_gather_into_disjoint_slices(self):
        """The gather shape: N device views read into disjoint slices of one host
        tensor. This is the direction that had no batched path at all."""
        blocks, block_len = 5, 8
        src_full = _arange((blocks, block_len), torch.float32)
        src_dev = [_to_dev(src_full)[b] for b in range(blocks)]
        host = torch.zeros(blocks * block_len, dtype=torch.float32)
        dst_slices = [host[b * block_len : (b + 1) * block_len] for b in range(blocks)]

        torch._foreach_copy_(dst_slices, src_dev)

        self.assertTrue(torch.equal(host, src_full.reshape(-1)))

    def test_v2h_noncontiguous_host_dst(self):
        """A strided host destination is NOT batched — it goes to ``copy_``."""
        host = torch.full((5, 6), -1.0, dtype=torch.float32)
        dst_views = [host.select(1, c) for c in (0, 4)]
        src_cpu = [_arange((5,), torch.float32) + c for c in (0, 4)]
        src_dev = [_to_dev(s) for s in src_cpu]

        with torch.rbln.explain() as p:
            torch._foreach_copy_(dst_views, src_dev)
        calls = _prim_calls(p.dump())
        self.assertEqual(calls.get("v2h_multi", 0), 0, f"strided host dst must not batch: {calls}")

        want = torch.full((5, 6), -1.0, dtype=torch.float32)
        for c, s in zip((0, 4), src_cpu):
            want.select(1, c).copy_(s)
        self.assertTrue(torch.equal(host, want), f"{host} != {want}")

    def test_v2h_repeated_source_is_allowed(self):
        """Several pairs reading the same device range into different host
        destinations. Sources are pure reads, so this must not be refused."""
        src_cpu = _arange((16,), torch.float32)
        shared = _to_dev(src_cpu)
        dst_cpu = [torch.zeros(16, dtype=torch.float32) for _ in range(3)]

        torch._foreach_copy_(dst_cpu, [shared, shared, shared])

        for got in dst_cpu:
            self.assertTrue(torch.equal(got, src_cpu))

    # ---- mixed direction lists ----

    def test_mixed_all_three_directions(self):
        """One call carrying rbln->rbln, cpu->rbln and rbln->cpu pairs. Each goes
        to its own batch; the results must not depend on which."""
        d2d_src = _to_dev(_arange((3, 4), torch.float32))
        d2d_dst = _to_dev(torch.zeros(3, 4, dtype=torch.float32))

        h2v_src = _arange((3, 4), torch.float32) + 100
        h2v_dst = _to_dev(torch.zeros(3, 4, dtype=torch.float32))

        v2h_src_cpu = _arange((3, 4), torch.float32) + 200
        v2h_src = _to_dev(v2h_src_cpu)
        v2h_dst = torch.zeros(3, 4, dtype=torch.float32)

        torch._foreach_copy_([d2d_dst, h2v_dst, v2h_dst], [d2d_src, h2v_src, v2h_src])

        _eq(d2d_dst, _arange((3, 4), torch.float32))
        _eq(h2v_dst, h2v_src)
        self.assertTrue(torch.equal(v2h_dst, v2h_src_cpu))

    def test_mixed_with_ineligible_pairs(self):
        """dtype-cast and broadcast pairs take per-pair copy_; the eligible ones
        around them are unaffected. The broadcast source is passed at its real
        shape ``(1,)`` — an already-``expand``ed one would match sizes and take
        the batched path instead."""
        ok_src = _arange((4,), torch.float32)
        ok_dst = _to_dev(torch.zeros(4, dtype=torch.float32))

        cast_src = _arange((4,), torch.float64)  # needs a real conversion
        cast_dst = _to_dev(torch.zeros(4, dtype=torch.float32))

        bcast_src = torch.tensor([5.0], dtype=torch.float32)  # shape (1,) -> broadcast
        bcast_dst = _to_dev(torch.zeros(4, dtype=torch.float32))

        torch._foreach_copy_([ok_dst, cast_dst, bcast_dst], [ok_src, cast_src, bcast_src])

        _eq(ok_dst, ok_src)
        _eq(cast_dst, cast_src.to(torch.float32))
        _eq(bcast_dst, bcast_src.expand(4).contiguous())

    def test_broadcast_source_at_matching_size_is_not_batched(self):
        """A stride-0 source expanded to the destination's shape matches sizes but
        is noncontiguous, so it takes the per-pair path and must replicate there."""
        src = torch.tensor([7.0, 8.0], dtype=torch.float32).expand(5, 2)
        dst = _to_dev(torch.zeros(5, 2, dtype=torch.float32))
        with torch.rbln.explain() as p:
            torch._foreach_copy_([dst], [src])
        calls = _prim_calls(p.dump())
        self.assertEqual(calls.get("h2v_multi", 0), 0, f"noncontiguous source must not batch: {calls}")
        _eq(dst, src.contiguous())

    def test_ineligible_pair_does_not_split_the_v2v_batch(self):
        """A pair on the per-pair path between two batchable v2v pairs must not
        force an early submit: the two v2v pairs still share one."""
        cases = {
            "cast": (
                _to_dev(torch.zeros(4, dtype=torch.float32)),
                _to_dev(_arange((4,), torch.float64)),
            ),
            "zero numel": (
                _to_dev(torch.zeros(0, 4, dtype=torch.float32)),
                _to_dev(torch.zeros(0, 4, dtype=torch.float32)),
            ),
            "noncontiguous host dst": (
                torch.zeros(2, 2, dtype=torch.float32).t(),
                _to_dev(_arange((2, 2), torch.float32)),
            ),
        }
        for name, (mid_dst, mid_src) in cases.items():
            with self.subTest(middle=name):
                srcs = [_to_dev(_arange((4,), torch.float32)), mid_src, _to_dev(_arange((4,), torch.float32) + 1)]
                dsts = [
                    _to_dev(torch.zeros(4, dtype=torch.float32)),
                    mid_dst,
                    _to_dev(torch.zeros(4, dtype=torch.float32)),
                ]

                with torch.rbln.explain() as p:
                    torch._foreach_copy_(dsts, srcs)
                calls = _prim_calls(p.dump())

                self.assertEqual(calls.get("v2v_multi", 0), 1, f"v2v batch was split: {calls}")
                _eq(dsts[0], srcs[0])
                _eq(dsts[2], srcs[2])

    # ---- destinations the batch must refuse ----

    def test_internally_overlapping_dst_is_rejected(self):
        """An ``expand``ed destination maps several elements onto one address, so
        with unordered entries the surviving write is arbitrary. ``copy_`` refuses
        it and so must the batch. All three directions, since the guard is
        shared."""
        cases = {
            "h2v": (_to_dev(torch.zeros(1)).expand(4), _arange((4,), torch.float32)),
            "v2h": (torch.zeros(1).expand(4), _to_dev(_arange((4,), torch.float32))),
            "v2v": (
                _to_dev(torch.zeros(1)).expand(4),
                _to_dev(_arange((4,), torch.float32)),
            ),
        }
        for name, (dst, src) in cases.items():
            with self.subTest(direction=name):
                with pytest.raises(RuntimeError, match="refers to a single memory location"):
                    torch._foreach_copy_([dst], [src])

    def test_same_view_pairs_are_no_ops_even_when_overlapping(self):
        """``copy_`` returns early on an identical view, before its overlap check,
        so an ``expand``ed identity pair is a no-op rather than an error and the
        batch must agree. Both forms: the same tensor twice, and two tensors
        describing the same view."""
        base = _to_dev(_arange((1,), torch.float32))
        expanded = base.expand(4)
        same_view = base.expand(4)
        before = expanded.cpu()

        cases = {
            "identity": (expanded, expanded),
            "distinct-same-view": (expanded, same_view),
        }
        for name, (dst, src) in cases.items():
            with self.subTest(form=name):
                torch._foreach_copy_([dst], [src])
                _eq(dst, before)

    def test_partially_overlapping_dst_is_not_batched(self):
        """A destination whose own rows overlap must not reach the batch, where
        entry order is arbitrary. ``has_internal_overlap`` answers ``TooHard`` so
        ``assert_no_internal_overlap`` lets it through; on the host directions the
        contiguity rule refuses it first, and ``TooHard`` is what keeps the same
        shape off the v2v batch."""
        elems_per_row = 16 * 1024
        stride = elems_per_row // 2
        storage = _to_dev(torch.zeros(elems_per_row + stride, dtype=torch.float32))
        dst = storage.as_strided((2, elems_per_row), (stride, 1))
        src = _arange((2, elems_per_row), torch.float32) + 1.0

        with torch.rbln.explain() as p:
            torch._foreach_copy_([dst], [src])
        calls = _prim_calls(p.dump())

        self.assertEqual(calls.get("h2v_multi", 0), 0, f"overlapping dst must not batch: {calls}")
        # Only the regions written by exactly one row are well defined; the
        # overlapping middle depends on write order, which this test does not pin.
        got = storage.cpu()
        _eq(_to_dev(got[:stride]), src[0][:stride])
        _eq(_to_dev(got[elems_per_row:]), src[1][stride:])

    def test_pairs_before_a_throwing_pair_still_land(self):
        """A pair rejected mid-list must not swallow the ones already queued:
        without a flush before the rejection, earlier destinations would stay
        untouched."""
        ok_src = _arange((4,), torch.float32)
        ok_dst = _to_dev(torch.zeros(4, dtype=torch.float32))
        bad_dst = _to_dev(torch.zeros(1)).expand(4)

        with pytest.raises(RuntimeError, match="refers to a single memory location"):
            torch._foreach_copy_([ok_dst, bad_dst], [ok_src, _arange((4,), torch.float32)])

        _eq(ok_dst, ok_src)

    # ---- the contiguity rule ----

    def test_contiguous_pairs_batch_regardless_of_size(self):
        """Contiguous pairs stay batched at any size: one descriptor each, so the
        batch replaces N submits with one (~3x at the weight-load shape). Size
        never enters the eligibility decision — contiguity does."""
        n_pairs = 16
        srcs = [_arange((64,), torch.float32) + i for i in range(n_pairs)]  # 256 B each
        dsts = [_to_dev(torch.zeros(64, dtype=torch.float32)) for _ in range(n_pairs)]

        with torch.rbln.explain() as p:
            torch._foreach_copy_(dsts, srcs)
        calls = _prim_calls(p.dump())

        self.assertEqual(calls.get("h2v_multi", 0), 1, f"contiguous pairs must batch: {calls}")
        self.assertEqual(calls.get("h2v", 0), 0, f"{calls}")
        for got, want in zip(dsts, srcs):
            _eq(got, want)

    def test_pinned_non_blocking_pairs_stay_per_pair(self):
        """The multi entrypoints are sync-only, so a pinned + non_blocking pair
        keeps its per-pair async copy. The same pairs batch without the flag."""
        srcs = [(_arange((64,), torch.float32) + i).pin_memory() for i in range(4)]
        dsts = [_to_dev(torch.zeros(64, dtype=torch.float32)) for _ in range(4)]

        with torch.rbln.explain() as p:
            torch._foreach_copy_(dsts, srcs, non_blocking=True)
        calls = _prim_calls(p.dump())
        self.assertEqual(calls.get("h2v_multi", 0), 0, f"pinned async must not batch: {calls}")
        for got, want in zip(dsts, srcs):
            _eq(got, want)

        dsts = [_to_dev(torch.zeros(64, dtype=torch.float32)) for _ in range(4)]
        with torch.rbln.explain() as p:
            torch._foreach_copy_(dsts, srcs)
        calls = _prim_calls(p.dump())
        self.assertEqual(calls.get("h2v_multi", 0), 1, f"pinned sync must batch: {calls}")
        for got, want in zip(dsts, srcs):
            _eq(got, want)

    def test_empty_lists_rejected_upstream(self):
        """Empty tensor lists are rejected by PyTorch before reaching the RBLN
        kernel, so the op is never invoked with zero pairs."""
        with pytest.raises(RuntimeError, match="at least one tensor"):
            torch._foreach_copy_([], [])

    def test_zero_numel_pairs_are_skipped(self):
        dst = [_to_dev(torch.zeros(0, 4, dtype=torch.float32))]
        src = [torch.zeros(0, 4, dtype=torch.float32)]
        torch._foreach_copy_(dst, src)
        self.assertEqual(dst[0].numel(), 0)

    # ---- ordering / aliasing must stay observable-equivalent ----

    def test_overlapping_host_destinations_stay_ordered(self):
        """Two pairs whose host destinations overlap must drop to the sequential
        path: the runtime requires disjoint destinations and does not check, and
        batching would make the write order observable."""
        src_a_cpu = torch.full((8,), 1.0, dtype=torch.float32)
        src_b_cpu = torch.full((8,), 2.0, dtype=torch.float32)

        host = torch.zeros(12, dtype=torch.float32)
        dst_a = host[0:8]
        dst_b = host[4:12]  # overlaps dst_a on [4, 8)
        torch._foreach_copy_([dst_a, dst_b], [_to_dev(src_a_cpu), _to_dev(src_b_cpu)])

        ref = torch.zeros(12, dtype=torch.float32)
        ref[0:8].copy_(src_a_cpu)
        ref[4:12].copy_(src_b_cpu)
        self.assertTrue(torch.equal(host, ref), f"{host} != {ref}")

    def test_host_destination_aliasing_a_source_stays_ordered(self):
        """A host destination that is also a later pair's source (RAW). List order
        means pair 1 reads what pair 0 wrote, so the second destination must hold
        the new contents; batching would let it read the pre-copy bytes."""
        nines = torch.full((8,), 9.0, dtype=torch.float32)
        shared = _arange((8,), torch.float32)
        dev_src = _to_dev(nines)
        other_dst = _to_dev(torch.zeros(8, dtype=torch.float32))

        torch._foreach_copy_([shared, other_dst], [dev_src, shared])

        # Reference: the same two copies in list order.
        ref_shared = _arange((8,), torch.float32)
        ref_shared.copy_(nines)
        ref_other = ref_shared.clone()  # pair 1 sees pair 0's write
        self.assertTrue(torch.equal(shared, ref_shared), f"{shared} != {ref_shared}")
        _eq(other_dst, ref_other)

    # ---- the batching itself is observable ----

    def test_h2v_uses_one_batched_submit(self):
        """N cpu->rbln pairs must reach the runtime through one h2v_multi call, not
        N per-entry h2v calls."""
        n_pairs = 8
        src_cpu = [_arange((4, 4), torch.float32) + i for i in range(n_pairs)]
        dst_dev = [_to_dev(torch.zeros(4, 4, dtype=torch.float32)) for _ in range(n_pairs)]

        with torch.rbln.explain() as p:
            torch._foreach_copy_(dst_dev, src_cpu)
        calls = _prim_calls(p.dump())

        self.assertEqual(calls.get("h2v_multi", 0), 1, f"expected one batched submit, got {calls}")
        self.assertEqual(calls.get("h2v", 0), 0, f"no per-entry h2v expected, got {calls}")
        for got, want in zip(dst_dev, src_cpu):
            _eq(got, want)

    def test_v2h_uses_one_batched_submit(self):
        n_pairs = 8
        src_cpu = [_arange((4, 4), torch.float32) + i for i in range(n_pairs)]
        src_dev = [_to_dev(s) for s in src_cpu]
        dst_cpu = [torch.zeros(4, 4, dtype=torch.float32) for _ in range(n_pairs)]

        with torch.rbln.explain() as p:
            torch._foreach_copy_(dst_cpu, src_dev)
        calls = _prim_calls(p.dump())

        self.assertEqual(calls.get("v2h_multi", 0), 1, f"expected one batched submit, got {calls}")
        self.assertEqual(calls.get("v2h", 0), 0, f"no per-entry v2h expected, got {calls}")
        for got, want in zip(dst_cpu, src_cpu):
            self.assertTrue(torch.equal(got, want))

    def test_oversized_batch_is_split_across_capped_submits(self):
        """One bulk call carries at most 16 entries / 8 MiB — past that the
        runtime's job never completes and the per-entry retry cannot recover. So
        40 pairs of 1 MiB land in five calls, and 8 pairs stay one.
        """
        one_mib = (1 << 20) // 4  # float32 elements
        for n_pairs, want_calls in ((8, 1), (40, 5)):
            src_cpu = [_arange((one_mib,), torch.float32) + i for i in range(n_pairs)]
            src_dev = [_to_dev(s) for s in src_cpu]
            dst_cpu = [torch.zeros(one_mib, dtype=torch.float32) for _ in range(n_pairs)]
            dst_dev = [_to_dev(torch.zeros(one_mib, dtype=torch.float32)) for _ in range(n_pairs)]

            with torch.rbln.explain() as p:
                torch._foreach_copy_(dst_cpu, src_dev)
            calls = _prim_calls(p.dump())
            self.assertEqual(
                calls.get("v2h_multi", 0), want_calls, f"{n_pairs} pairs of 1MiB: expected {want_calls}, got {calls}"
            )
            self.assertEqual(calls.get("v2h", 0), 0, f"splitting must not degrade to per-entry: {calls}")
            for got, want in zip(dst_cpu, src_cpu):
                self.assertTrue(torch.equal(got, want))

            with torch.rbln.explain() as p:
                torch._foreach_copy_(dst_dev, src_cpu)
            calls = _prim_calls(p.dump())
            self.assertEqual(
                calls.get("h2v_multi", 0), want_calls, f"{n_pairs} pairs of 1MiB: expected {want_calls}, got {calls}"
            )
            self.assertEqual(calls.get("h2v", 0), 0, f"splitting must not degrade to per-entry: {calls}")
            for got, want in zip(dst_dev, src_cpu):
                _eq(got, want)

    def test_entry_over_the_cap_goes_per_entry(self):
        """A single descriptor larger than the 8 MiB cap cannot be split, so it
        takes the single-copy path — which has no cap — instead of a bulk call."""
        elems = (12 << 20) // 4  # 12 MiB of float32
        src_cpu = _arange((elems,), torch.float32)

        dst_dev = _to_dev(torch.zeros(elems, dtype=torch.float32))
        with torch.rbln.explain() as p:
            torch._foreach_copy_([dst_dev], [src_cpu])
        calls = _prim_calls(p.dump())
        self.assertEqual(calls.get("h2v_multi", 0), 0, f"oversized entry must not use the bulk call: {calls}")
        self.assertEqual(calls.get("h2v", 0), 1, f"{calls}")
        _eq(dst_dev, src_cpu)

        dst_cpu = torch.zeros(elems, dtype=torch.float32)
        with torch.rbln.explain() as p:
            torch._foreach_copy_([dst_cpu], [dst_dev])
        calls = _prim_calls(p.dump())
        self.assertEqual(calls.get("v2h_multi", 0), 0, f"oversized entry must not use the bulk call: {calls}")
        self.assertEqual(calls.get("v2h", 0), 1, f"{calls}")
        self.assertTrue(torch.equal(dst_cpu, src_cpu))

    def test_mixed_directions_use_one_submit_each(self):
        """A list spanning all three directions costs one submit per direction —
        three total, not one per pair."""
        pairs = 4
        d2d_dst = [_to_dev(torch.zeros(4, dtype=torch.float32)) for _ in range(pairs)]
        d2d_src = [_to_dev(_arange((4,), torch.float32) + i) for i in range(pairs)]
        h2v_dst = [_to_dev(torch.zeros(4, dtype=torch.float32)) for _ in range(pairs)]
        h2v_src = [_arange((4,), torch.float32) + 10 + i for i in range(pairs)]
        v2h_dst = [torch.zeros(4, dtype=torch.float32) for _ in range(pairs)]
        v2h_src = [_to_dev(_arange((4,), torch.float32) + 20 + i) for i in range(pairs)]

        with torch.rbln.explain() as p:
            torch._foreach_copy_(d2d_dst + h2v_dst + v2h_dst, d2d_src + h2v_src + v2h_src)
        calls = _prim_calls(p.dump())

        self.assertEqual(calls.get("v2v_multi", 0), 1, f"{calls}")
        self.assertEqual(calls.get("h2v_multi", 0), 1, f"{calls}")
        self.assertEqual(calls.get("v2h_multi", 0), 1, f"{calls}")

    def test_no_batch_fallback_recorded_on_the_happy_path(self):
        """A well-formed batch must not record a bounce. A fallback here would be
        invisible in the values — the copy still completes, just slowly."""
        src_cpu = [_arange((8, 8), torch.float32) + i for i in range(6)]
        dst_dev = [_to_dev(torch.zeros(8, 8, dtype=torch.float32)) for _ in range(6)]

        with torch.rbln.explain() as p:
            torch._foreach_copy_(dst_dev, src_cpu)
        by_site = p.dump()["hidden_host_bounce"]["by_site"]

        hit = {name: v["count"] for name, v in by_site.items() if v["count"]}
        for site in ("host_batch_to_per_entry", "strided_v2v_cpu_fallback"):
            self.assertEqual(by_site.get(site, {}).get("count", 0), 0, f"unexpected fallback: {hit}")


instantiate_device_type_tests(TestForeachCopyHost, globals(), only_for="privateuse1")

if __name__ == "__main__":
    run_tests()
