# Owner(s): ["module: PrivateUse1"]

"""``copy_`` from a strided device source, served by one compiled device program.

A device->device copy whose source is a view had two routes: the strided v2v engine,
which spends a descriptor per contiguous run, and above its cap a host round-trip.
Neither fits a copy that is one reshape away from contiguous -- a KV block read
head-major and written token-major breaks into runs of a single head_size row, far past
the cap. Replaying the view inside a compiled graph moves it in one program instead.

Which route ran is observable, and it takes two counters, not one. Every shape here is
past the strided engine's cap, so without the program the copy round-trips the host and
the profiler records a bounce -- that separates "served" from "declined", which the
result cannot, since the slow routes are correct too. But a served copy can still reach
the host: below the compiler's last-dim alignment the program is lowered with its input
on the host, and the runtime feeds it there without any bounce being recorded. So the
served cases also assert that the runtime ran no host primitive.
"""

from __future__ import annotations

import pytest
import torch
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase

from test.utils import requires_logical_devices
from test.utils_v2v import to_dev as _to_dev


# Past ::rbln::kMaxV2VMultiCopies once permuted: the strided engine declines and the
# alternative is a host bounce, which is what makes the route observable.
_ROWS, _HEADS, _BLOCK, _DIM = 2, 4, 256, 128


_HOST_PRIMITIVES = ("v2h", "h2v")


def _bounces(report: dict) -> int:
    return int(report["hidden_host_bounce"]["total_count"])


def _host_primitives(report: dict) -> dict:
    """The runtime's own host transfers, which a bounce count does not cover."""
    by_primitive = report["rebel_runtime"]["by_primitive"]
    return {k: v for k, v in by_primitive.items() if k in _HOST_PRIMITIVES}


def _copy_and_count(dst: torch.Tensor, src: torch.Tensor) -> tuple[int, dict]:
    with torch.rbln.explain() as p:
        dst.copy_(src)
    report = p.dump()
    return _bounces(report), _host_primitives(report)


# View chains the detector classifies. Each maps a contiguous host tensor to the view
# whose copy is under test; the reference is the same chain taken on the host.
_CHAINS = {
    "permute": (
        (_ROWS, _HEADS, _BLOCK, _DIM),
        lambda t: t.permute(0, 2, 1, 3),
    ),
    # leading dims merged before the permute -- what a KV transfer writes
    "reshape_then_permute": (
        (2, _ROWS, _HEADS, _BLOCK, _DIM),
        lambda t: t.reshape(2 * _ROWS, _HEADS, _BLOCK, _DIM).permute(0, 2, 1, 3),
    ),
    # one dim split instead of merged
    "split_then_permute": (
        (_ROWS * _HEADS, _BLOCK, _DIM),
        lambda t: t.reshape(_ROWS, _HEADS, _BLOCK, _DIM).permute(0, 2, 1, 3),
    ),
    # a slice of a larger buffer: the recipe carries select and narrow too
    "select_then_permute": (
        (2, _ROWS, _HEADS, _BLOCK, _DIM),
        lambda t: t[1].permute(0, 2, 1, 3),
    ),
    "narrow_then_permute": (
        (2 * _ROWS, _HEADS, _BLOCK, _DIM),
        lambda t: t[:_ROWS].permute(0, 2, 1, 3),
    ),
    # not a permutation of the leading dims at all
    "transpose_last_two": (
        (_ROWS, _HEADS, _BLOCK, _DIM),
        lambda t: t.transpose(-1, -2),
    ),
}


@pytest.mark.test_set_ci
class TestViewCopy(TestCase):
    """Device->device ``copy_`` whose source is a view."""

    def _run(self, shape, view_fn, dtype=torch.bfloat16):
        """Returns (bounces, device result, host reference)."""
        host = torch.randint(-1000, 1000, shape).to(dtype)
        src = view_fn(_to_dev(host))
        dst = torch.empty(src.shape, dtype=dtype, device=src.device)
        bounces, host_primitives = _copy_and_count(dst, src)
        return (bounces, host_primitives), dst.cpu(), view_fn(host).contiguous()

    # ---- chains the program serves ----

    @parametrize("chain", sorted(_CHAINS))
    def test_a_classifiable_view_is_served_by_the_program(self, device, chain):
        shape, view_fn = _CHAINS[chain]
        (bounces, host_primitives), got, want = self._run(shape, view_fn)
        self.assertEqual(bounces, 0, f"{chain} must not round-trip the host")
        self.assertEqual(host_primitives, {}, f"{chain} ran on the host, not the device")
        self.assertTrue(torch.equal(got, want), f"{chain} returned different values")

    @parametrize("chain", sorted(_CHAINS))
    def test_a_repeated_copy_stays_on_the_program(self, device, chain):
        """The program is cached per view pattern; a repeat must not fall off it."""
        shape, view_fn = _CHAINS[chain]
        host = torch.randint(-1000, 1000, shape).to(torch.bfloat16)
        src = view_fn(_to_dev(host))
        dst = torch.empty(src.shape, dtype=torch.bfloat16, device=src.device)
        for i in range(3):
            bounces, host_primitives = _copy_and_count(dst, src)
            self.assertEqual(bounces, 0, f"{chain} bounced on call {i}")
            self.assertEqual(host_primitives, {}, f"{chain} reached the host on call {i}")
        self.assertTrue(torch.equal(dst.cpu(), view_fn(host).contiguous()))

    def test_the_detector_classifies_every_served_chain(self, device):
        """A route regression surfaces here first: once the detector stops classifying a
        chain, its copy drops back to the slow route and only the timing shows it."""
        from torch_rbln._internal.ops_utils import _detect_view_recipe

        for chain, (shape, view_fn) in sorted(_CHAINS.items()):
            view = view_fn(torch.empty(shape, dtype=torch.bfloat16))
            self.assertIsNotNone(_detect_view_recipe(view), f"{chain} is no longer classified")

    # ---- what stays on the existing route ----

    @dtypes(torch.float16, torch.float32, torch.int32, torch.bool)
    def test_a_dtype_the_device_rewrites_keeps_the_slow_route(self, device, dtype):
        """Only bf16 reaches the device as itself; the rest are rewritten to a narrower
        float, and ``copy_`` is not a compute op -- both routes have to agree."""
        shape, view_fn = _CHAINS["permute"]
        (bounces, _), got, want = self._run(shape, view_fn, dtype=dtype)
        self.assertGreater(bounces, 0, f"{dtype} must not take the program")
        self.assertTrue(torch.equal(got, want))

    @parametrize("last_dim", [96, 127, 1])
    def test_an_unaligned_last_dim_keeps_the_slow_route(self, device, last_dim):
        """Below the compiler's last-dim alignment the tensor stays on the host, where
        the program is orders of magnitude slower than the walk it would replace."""
        shape = (_ROWS, _HEADS, _BLOCK, last_dim)
        (bounces, _), got, want = self._run(shape, lambda t: t.permute(0, 2, 1, 3))
        self.assertGreater(bounces, 0, f"last dim {last_dim} must not take the program")
        self.assertTrue(torch.equal(got, want))

    def test_a_strided_destination_keeps_the_slow_route(self, device):
        """The program writes a contiguous buffer; scattering into a strided destination
        is not something the out-tensor path can express."""
        host = torch.randint(-1000, 1000, (_ROWS, _HEADS, _BLOCK, _DIM)).to(torch.bfloat16)
        src = _to_dev(host)
        dst = torch.empty((_ROWS, _BLOCK, _HEADS, _DIM), dtype=torch.bfloat16, device=src.device).permute(0, 2, 1, 3)
        bounces, _ = _copy_and_count(dst, src)
        self.assertGreater(bounces, 0, "a strided destination must not take the program")
        self.assertTrue(torch.equal(dst.contiguous().cpu(), host))

    def test_an_unclassifiable_view_keeps_the_slow_route(self, device):
        """The op has to answer before ``prepare_args_view_aware`` would reach for
        ``.contiguous()``, which re-enters ``copy_``."""
        host = torch.randint(-1000, 1000, (_ROWS, _HEADS, _BLOCK, _DIM)).to(torch.bfloat16)
        # Strides no chain of view ops produces: the middle one is not a product of the
        # sizes below it, so the rows overlap rather than tile.
        sizes, strides = (_ROWS, _HEADS, _BLOCK, _DIM), (100000, 20000, _DIM + 1, 1)
        src = _to_dev(host).as_strided(sizes, strides, 0)
        dst = torch.empty(sizes, dtype=torch.bfloat16, device=src.device)
        bounces, _ = _copy_and_count(dst, src)
        self.assertGreater(bounces, 0, "an unclassified view must not take the program")
        self.assertTrue(torch.equal(dst.cpu(), host.as_strided(sizes, strides, 0).contiguous()))

    def test_a_contiguous_copy_is_untouched(self, device):
        host = torch.randint(-1000, 1000, (_ROWS, _HEADS, _BLOCK, _DIM)).to(torch.bfloat16)
        src = _to_dev(host)
        dst = torch.empty_like(src)
        self.assertEqual(_copy_and_count(dst, src), (0, {}), "a direct copy must not bounce")
        self.assertTrue(torch.equal(dst.cpu(), host))

    def test_a_destination_at_a_storage_offset_is_served(self, device):
        """The compile path writes the caller's buffer only when it starts at offset 0;
        otherwise it hands back its own and the result is moved across. Still on device --
        the move is a v2v -- but it is the one branch where the two buffers differ."""
        host = torch.randint(-1000, 1000, (_ROWS, _HEADS, _BLOCK, _DIM)).to(torch.bfloat16)
        src = _to_dev(host).permute(0, 2, 1, 3)
        storage = torch.empty((2 * _ROWS, _BLOCK, _HEADS, _DIM), dtype=torch.bfloat16, device=src.device)
        dst = storage[_ROWS:]
        self.assertNotEqual(dst.storage_offset(), 0)
        bounces, host_primitives = _copy_and_count(dst, src)
        self.assertEqual(bounces, 0, "an offset destination must not round-trip the host")
        self.assertEqual(host_primitives, {}, "an offset destination must stay on device")
        self.assertTrue(torch.equal(dst.cpu(), host.permute(0, 2, 1, 3).contiguous()))

    @requires_logical_devices(2)
    def test_a_copy_across_devices_is_untouched(self, device):
        """One program reads and writes one device, so a copy that crosses them is not
        this path's -- and taking it anyway writes the wrong buffer, not just a slow one."""
        host = torch.randint(-1000, 1000, (_ROWS, _HEADS, _BLOCK, _DIM)).to(torch.bfloat16)
        src = host.to("rbln:0").permute(0, 2, 1, 3)
        dst = torch.empty(src.shape, dtype=torch.bfloat16, device="rbln:1")
        dst.copy_(src)
        torch.rbln.synchronize()
        self.assertTrue(torch.equal(dst.cpu(), host.permute(0, 2, 1, 3).contiguous()))

    def test_a_device_to_host_copy_is_untouched(self, device):
        host = torch.randint(-1000, 1000, (_ROWS, _HEADS, _BLOCK, _DIM)).to(torch.bfloat16)
        src = _to_dev(host).permute(0, 2, 1, 3)
        dst = torch.empty(src.shape, dtype=torch.bfloat16)
        dst.copy_(src)
        torch.rbln.synchronize()
        self.assertTrue(torch.equal(dst, host.permute(0, 2, 1, 3).contiguous()))


instantiate_device_type_tests(TestViewCopy, globals(), only_for="privateuse1")

if __name__ == "__main__":
    run_tests()


@pytest.mark.test_set_ci
class TestViewCopyInPlace(TestCase):
    """``torch_rbln::copy_strided_view(..., inplace=True)``: the program permutes a buffer
    in its own storage, so a caller that only needs the permuted layout holds one buffer,
    not two.

    ``copy_`` never reaches this op (ATen refuses partially overlapping pairs), so it is
    called directly, the way the LMCache transfer does. The first call for a geometry is
    checked against an out-of-place copy inside the op; these tests check the op against
    the host instead, so a passing verification that compared two equally wrong results
    would still be caught.
    """

    _SHAPE = (2 * _ROWS, _HEADS, _BLOCK, _DIM)  # rows, heads, tokens, head_size

    def _permute_in_place(self, host: torch.Tensor) -> torch.Tensor:
        rows, heads, block, dim = host.shape
        buf = _to_dev(host)
        src = buf.view(rows, heads, block, dim).permute(0, 2, 1, 3)
        out = buf.view(rows, block, heads, dim)
        self.assertTrue(torch.ops.torch_rbln.copy_strided_view(src, out, True))
        return buf

    def test_permutes_the_buffer_in_place(self, device):
        host = torch.randint(-1000, 1000, self._SHAPE).to(torch.bfloat16)
        buf = self._permute_in_place(host)
        rows, heads, block, dim = self._SHAPE
        self.assertEqual(buf.view(rows, block, heads, dim).cpu(), host.permute(0, 2, 1, 3))

    def test_repeated_calls_stay_exact_and_allocate_nothing(self, device):
        # After the geometry's one-time verification, a call must not grow device memory:
        # the program writes the caller's buffer, there is no output of its own to keep.
        host = torch.randint(-1000, 1000, self._SHAPE).to(torch.bfloat16)
        buf = self._permute_in_place(host)  # verifies the geometry
        rows, heads, block, dim = self._SHAPE
        for _ in range(3):
            host = torch.randint(-1000, 1000, self._SHAPE).to(torch.bfloat16)
            buf.copy_(host)
            torch.rbln.synchronize()
            before = torch.rbln.memory_allocated(buf.device)
            src = buf.view(rows, heads, block, dim).permute(0, 2, 1, 3)
            torch.ops.torch_rbln.copy_strided_view(src, buf.view(rows, block, heads, dim), True)
            torch.rbln.synchronize()
            self.assertEqual(torch.rbln.memory_allocated(buf.device), before)
            self.assertEqual(buf.view(rows, block, heads, dim).cpu(), host.permute(0, 2, 1, 3))

    def test_rejects_an_out_that_is_not_the_whole_buffer(self, device):
        rows, heads, block, dim = self._SHAPE
        buf = _to_dev(torch.zeros(self._SHAPE, dtype=torch.bfloat16))
        src = buf.view(rows, heads, block, dim).permute(0, 2, 1, 3)
        other = torch.empty(rows, block, heads, dim, dtype=torch.bfloat16, device=buf.device)
        with self.assertRaisesRegex(RuntimeError, "whole buffer"):
            torch.ops.torch_rbln.copy_strided_view(src, other, True)
        with self.assertRaisesRegex(RuntimeError, "whole buffer"):
            torch.ops.torch_rbln.copy_strided_view(src[:1], buf.view(rows, block, heads, dim)[:1], True)

    def test_declines_an_unclassifiable_view(self, device):
        # Same contract as copy_strided_view: False, and the buffer untouched.
        sizes = (_ROWS, _HEADS, _BLOCK, _DIM)
        host = torch.randint(-1000, 1000, sizes).to(torch.bfloat16)
        buf = _to_dev(host)
        # Strides no chain of view ops produces (see the copy_ suite above).
        src = buf.as_strided(sizes, (100000, 20000, _DIM + 1, 1), 0)
        self.assertFalse(torch.ops.torch_rbln.copy_strided_view(src, buf, True))
        self.assertEqual(buf.cpu(), host)


@pytest.mark.test_set_ci
@pytest.mark.usefixtures("enable_deploy_mode")
class TestViewCopyProgramSlots(TestCase):
    """One compiled program per source address, least recently used evicted past the cap.

    Rebinding a program to another buffer re-uploads its instruction streams through the
    allocator, which waits for every transfer in flight on the device first -- so two live
    buffers sharing a program would rebind on every call and stall a transfer pipeline on
    each one. The policy is checked through the module's slot table: the copies themselves
    are correct under any policy.
    """

    _SHAPE = (_ROWS, _HEADS, _BLOCK, _DIM)

    def setUp(self):
        super().setUp()
        # Third Party
        from torch_rbln._internal import register_custom_ops as rco

        self._rco = rco
        self._saved = (rco._MAX_COPY_SLOTS, dict(rco._copy_slot_of_src))

    def tearDown(self):
        rco = self._rco
        rco._MAX_COPY_SLOTS = self._saved[0]
        rco._copy_slot_of_src.clear()
        rco._copy_slot_of_src.update(self._saved[1])
        super().tearDown()

    def _copy(self, buf: torch.Tensor) -> None:
        out = torch.empty(_ROWS, _BLOCK, _HEADS, _DIM, dtype=buf.dtype, device=buf.device)
        out.copy_(buf.permute(0, 2, 1, 3))
        self.assertEqual(out.cpu(), buf.cpu().permute(0, 2, 1, 3))

    def test_live_buffers_within_the_cap_keep_their_own_slot(self, device):
        rco = self._rco
        rco._MAX_COPY_SLOTS = 4
        rco._copy_slot_of_src.clear()
        bufs = [_to_dev(torch.randint(-1000, 1000, self._SHAPE).to(torch.bfloat16)) for _ in range(4)]
        for _ in range(3):  # alternate over the whole set, as a pipeline over its staging does
            for buf in bufs:
                self._copy(buf)
        slots = [rco._copy_slot_of_src[b.data_ptr()] for b in bufs]
        self.assertEqual(sorted(slots), [0, 1, 2, 3])  # nobody moved, nobody shared

    def test_past_the_cap_the_least_recently_used_address_gives_up_its_slot(self, device):
        rco = self._rco
        rco._MAX_COPY_SLOTS = 3
        rco._copy_slot_of_src.clear()
        a, b, c, d = [_to_dev(torch.randint(-1000, 1000, self._SHAPE).to(torch.bfloat16)) for _ in range(4)]
        for buf in (a, b, c):
            self._copy(buf)
        self._copy(a)  # a is now the most recent; b the least
        slot_b = rco._copy_slot_of_src[b.data_ptr()]
        self._copy(d)
        self.assertNotIn(b.data_ptr(), rco._copy_slot_of_src)  # b evicted
        self.assertEqual(rco._copy_slot_of_src[d.data_ptr()], slot_b)  # d took b's slot
        self.assertIn(a.data_ptr(), rco._copy_slot_of_src)
        self.assertEqual(len(rco._copy_slot_of_src), 3)
        self._copy(b)  # b comes back onto a slot of its own and still copies right
        self.assertEqual(len(rco._copy_slot_of_src), 3)


instantiate_device_type_tests(TestViewCopyInPlace, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestViewCopyProgramSlots, globals(), only_for="privateuse1")
