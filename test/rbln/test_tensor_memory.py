# Owner(s): ["module: PrivateUse1"]

"""Test suite for tensor memory correctness.

Validates that device operations produce correct results under non-trivial memory
scenarios:
- Storage aliasing: multiple tensors sharing the same underlying storage
- Input/output memory independence: ensuring that input and output tensors do not share
  overlapping memory
- Bound allocations: device buffers handed to a consumer that DMAs out of band
- Huge host buffers: the host side of the same transfers
- Host registration: caller-owned host ranges pinned once for those transfers
"""

import ctypes
import gc
import mmap
import os
import sys
import weakref
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.shared_memory import SharedMemory
from typing import ClassVar
from unittest import mock

import pytest
import torch
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase

import torch_rbln._C
import torch_rbln.memory
from test.utils import run_in_isolated_process, SUPPORTED_DTYPES
from torch_rbln import _host_registration as admission


# Tolerance for numerical comparisons
ATOL = 0.01
RTOL = 0.01


@pytest.mark.npu(count=1)
class TestAliasedTensors(TestCase):
    binary_ops: ClassVar[list[Callable[..., torch.Tensor]]] = [torch.add]
    rbln_device = torch.device("rbln:0")
    shapes: ClassVar[list[tuple[int, int]]] = [(2, 16), (2, 64)]

    def _run_binary_op(self, binary_op, rbln_input, rbln_other):
        self.assertEqual(rbln_input.device.type, "rbln")
        self.assertEqual(rbln_other.device.type, "rbln")

        rbln_out = binary_op(rbln_input, rbln_other)

        cpu_input = rbln_input.cpu()
        cpu_other = rbln_other.cpu()
        cpu_out = binary_op(cpu_input, cpu_other)
        self.assertEqual(rbln_out.cpu(), cpu_out, atol=ATOL, rtol=RTOL)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("binary_op", binary_ops)
    @parametrize("shape", shapes)
    def test_same_base_tensor(self, dtype, binary_op, shape):
        x_base = torch.randn(shape, dtype=dtype, device=self.rbln_device)

        input = x_base
        other = x_base
        self.assertIs(input, other)
        self.assertEqual(input.data_ptr(), other.data_ptr())
        self._run_binary_op(binary_op, input, other)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("binary_op", binary_ops)
    @parametrize("shape", shapes)
    def test_different_base_tensors(self, dtype, binary_op, shape):
        x_base = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        y_base = torch.randn(shape, dtype=dtype, device=self.rbln_device)

        input = x_base
        other = y_base
        self.assertIsNot(input, other)
        self.assertNotEqual(input.data_ptr(), other.data_ptr())
        self._run_binary_op(binary_op, input, other)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("binary_op", binary_ops)
    @parametrize("shape", shapes)
    def test_same_view_tensor_from_same_base_tensor(self, dtype, binary_op, shape):
        x_base = torch.randn(shape, dtype=dtype, device=self.rbln_device)

        x_view = x_base.view(shape)
        self.assertEqual(x_view.size(), x_base.size())
        self.assertEqual(x_view.stride(), x_base.stride())
        self.assertEqual(x_view.storage_offset(), x_base.storage_offset())
        self.assertEqual(x_view.data_ptr(), x_base.data_ptr())

        input = x_view
        other = x_view
        self.assertIs(input, other)
        self.assertEqual(input.data_ptr(), other.data_ptr())
        self._run_binary_op(binary_op, input, other)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("binary_op", binary_ops)
    @parametrize("shape", shapes)
    def test_different_view_tensors_from_same_base_tensor(self, dtype, binary_op, shape):
        x_base = torch.randn(shape, dtype=dtype, device=self.rbln_device)

        x_view = x_base.view(shape)
        self.assertEqual(x_view.size(), x_base.size())
        self.assertEqual(x_view.stride(), x_base.stride())
        self.assertEqual(x_view.storage_offset(), x_base.storage_offset())
        self.assertEqual(x_view.data_ptr(), x_base.data_ptr())
        new_x_view = x_base.view(shape)
        self.assertEqual(new_x_view.size(), x_base.size())
        self.assertEqual(new_x_view.stride(), x_base.stride())
        self.assertEqual(new_x_view.storage_offset(), x_base.storage_offset())
        self.assertEqual(new_x_view.data_ptr(), x_base.data_ptr())

        input = x_view
        other = new_x_view
        self.assertIsNot(input, other)
        self.assertEqual(input.data_ptr(), other.data_ptr())
        self._run_binary_op(binary_op, input, other)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("binary_op", binary_ops)
    @parametrize("shape", shapes)
    def test_different_view_tensors_from_different_base_tensors(self, dtype, binary_op, shape):
        x_base = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        y_base = torch.randn(shape, dtype=dtype, device=self.rbln_device)

        x_view = x_base.view(shape)
        self.assertEqual(x_view.size(), x_base.size())
        self.assertEqual(x_view.stride(), x_base.stride())
        self.assertEqual(x_view.storage_offset(), x_base.storage_offset())
        self.assertEqual(x_view.data_ptr(), x_base.data_ptr())
        y_view = y_base.view(shape)
        self.assertEqual(y_view.size(), y_base.size())
        self.assertEqual(y_view.stride(), y_base.stride())
        self.assertEqual(y_view.storage_offset(), y_base.storage_offset())
        self.assertEqual(y_view.data_ptr(), y_base.data_ptr())

        input = x_view
        other = y_view
        self.assertIsNot(input, other)
        self.assertNotEqual(input.data_ptr(), other.data_ptr())
        self._run_binary_op(binary_op, input, other)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("binary_op", binary_ops)
    @parametrize("shape", shapes)
    def test_mixed_base_and_view_tensors(self, dtype, binary_op, shape):
        x_base = torch.randn(shape, dtype=dtype, device=self.rbln_device)

        x_view = x_base.view(shape)
        self.assertEqual(x_view.size(), x_base.size())
        self.assertEqual(x_view.stride(), x_base.stride())
        self.assertEqual(x_view.storage_offset(), x_base.storage_offset())
        self.assertEqual(x_view.data_ptr(), x_base.data_ptr())

        input = x_base
        other = x_view
        self.assertIsNot(input, other)
        self.assertEqual(input.data_ptr(), other.data_ptr())
        self._run_binary_op(binary_op, input, other)


def _input_output_tensor_memory_independence_worker(rbln_device, dtype):
    """Worker function that runs in a spawned subprocess to ensure a clean data_ptr
    counter.
    """
    # Create a seed input tensor whose internal key value may collide with output tensor
    # memory keys.
    seed_tensor = torch.randn([2, 2], dtype=dtype, device=rbln_device)
    seed_data_ptr = seed_tensor.data_ptr()

    rbln_out = torch.abs(seed_tensor)  # First output tensor allocation.
    cpu_out = torch.abs(seed_tensor.cpu())
    torch.testing.assert_close(rbln_out.cpu(), cpu_out, atol=ATOL, rtol=RTOL)

    # Burn through (seed_data_ptr - 1) output tensor allocations so the next output
    # tensor receives a key equal to seed_data_ptr, forcing a key collision.
    for i in range(seed_data_ptr - 1):
        # Vary the size of the tensor to ensure unique memory allocation.
        t = torch.randn([i], dtype=dtype, device=rbln_device)
        rbln_out = torch.abs(t)
        cpu_out = torch.abs(t.cpu())
        torch.testing.assert_close(rbln_out.cpu(), cpu_out, atol=ATOL, rtol=RTOL)
    # After the loop, the next output tensor key may collide with the seed tensor key.

    x = torch.randn([2, 2, 4], dtype=dtype, device=rbln_device)
    y = torch.randn([2, 2, 4], dtype=dtype, device=rbln_device)

    # Use trunc-mode division to amplify numerical divergence if the backend silently
    # returns wrong data.
    rbln_out = torch.div(x, y, rounding_mode="trunc")
    cpu_out = torch.div(x.cpu(), y.cpu(), rounding_mode="trunc")
    torch.testing.assert_close(rbln_out.cpu(), cpu_out, atol=ATOL, rtol=RTOL)


@pytest.mark.npu(count=1)
class TestInputOutputTensors(TestCase):
    """Regression tests for input and output tensor memory collisions.

    The RBLN backend may confuse an output tensor with an already-allocated
    input tensor, silently producing incorrect results. This test reproduces
    such a pattern and verifies that the results remain correct.

    The test runs in a spawned subprocess so the data_ptr counter starts from
    a clean state regardless of how many other tests have run before it. This
    is because when data_ptr is too large, it takes a long time to reach the
    collision point, making the test impractically slow.
    """

    rbln_device = torch.device("rbln:0")

    @dtypes(*SUPPORTED_DTYPES)
    def test_input_output_tensor_memory_independence(self, dtype):
        run_in_isolated_process(_input_output_tensor_memory_independence_worker, self.rbln_device, dtype)


def _device_bytes() -> int:
    """Bytes of device memory the runtime holds right now.

    Process-global and recorded at every runtime buffer alloc/free, so only a delta
    taken across the call under test says anything.
    """
    return torch_rbln._C._rt_prof_memory()[0]


def _bind_after_shutdown_worker(device):
    """Assert ``bind_device_memory`` raises rather than faulting once the runtime is
    down.

    Runs in a spawned process: the shutdown flag is process-global and set-once, so
    flipping it here would poison every later test on this worker.
    """
    t = torch.empty(4096, dtype=torch.uint8, device=device)
    torch_rbln._C._set_runtime_shutting_down(shutting_down=True)
    try:
        torch.rbln.bind_device_memory(t)
    except RuntimeError:
        return
    msg = "bind_device_memory did not raise with the runtime shutting down"
    raise AssertionError(msg)


@pytest.mark.npu(count=1)
class TestBindDeviceMemory(TestCase):
    """``torch.rbln.bind_device_memory`` materializes a tensor's device allocation, for
    a consumer that reads the physical buffers out of band and so never runs the op
    that would otherwise trigger the lazy bind.

    The runtime's device-memory gauge is what makes that observable from Python; no
    torch op stands in for it, since an op either materializes the allocation itself
    or -- ``zero_`` -- marks it logically zero without allocating anything.
    """

    rbln_device = torch.device("rbln:0")

    def test_bind_allocates_the_physical_memory(self):
        # The postcondition itself. Nothing but the bind can account for the delta:
        # the allocation is still lazy (asserted), the device and its one-time pools
        # are already committed by the synchronize, and no op runs on ``t``.
        nbytes = 8 << 20
        with mock.patch.dict(os.environ):
            os.environ.pop("TORCH_RBLN_EAGER_MALLOC", None)  # would bind at allocation
            torch.rbln.synchronize()

            before_alloc = _device_bytes()
            t = torch.empty(nbytes, dtype=torch.uint8, device=self.rbln_device)
            before_bind = _device_bytes()
            torch.rbln.bind_device_memory(t)
            after_bind = _device_bytes()

        self.assertEqual(before_bind, before_alloc, "the allocation was expected to still be lazy")
        # The runtime may bring more along -- a slab it grew to serve this -- but not
        # less.
        self.assertGreaterEqual(after_bind - before_bind, nbytes)

    def test_binds_a_whole_tensor(self):
        t = torch.empty(4096, dtype=torch.uint8, device=self.rbln_device)
        torch.rbln.bind_device_memory(t)
        # The region must still behave like any other device tensor afterwards. Not
        # ``zero_``: that marks the allocation logically zero, and the next device read
        # then serves zeros over whatever a consumer wrote into it out of band.
        payload = torch.arange(4096, dtype=torch.int32).to(torch.uint8)
        t.copy_(payload)
        self.assertEqual(t.cpu(), payload)

    def test_rebinding_is_allowed(self):
        # The collective path re-binds the same tensor before every operation, so a
        # second bind must not fail -- and must not allocate the region a second time.
        nbytes = 8 << 20
        t = torch.empty(nbytes, dtype=torch.uint8, device=self.rbln_device)
        torch.rbln.bind_device_memory(t)
        before = _device_bytes()
        torch.rbln.bind_device_memory(t)
        self.assertLess(_device_bytes() - before, nbytes)

    def test_rejects_cpu_tensor(self):
        with self.assertRaises(RuntimeError):
            torch.rbln.bind_device_memory(torch.empty(4096, dtype=torch.uint8))

    def test_rejects_interior_view(self):
        # A slice's data_ptr() is an interior address; binding it would configure the
        # wrong region, so it is refused rather than silently mis-bound. A view that
        # still spans the whole storage carries the allocation's own address and size,
        # and is accepted.
        base = torch.empty(4096, dtype=torch.uint8, device=self.rbln_device)
        with self.assertRaises(RuntimeError):
            torch.rbln.bind_device_memory(base[16:])
        torch.rbln.bind_device_memory(base.view(64, 64))

    @pytest.mark.serial
    def test_raises_once_runtime_is_shutting_down(self):
        # The raw runtime call would fault rather than raise past teardown, and this
        # entry point is public, so a caller can reach it there.
        run_in_isolated_process(_bind_after_shutdown_worker, str(self.rbln_device))


class TestHugeHostEmpty(TestCase):
    """``torch.rbln.huge_host_empty`` returns host memory the device can DMA into
    without staging through a bounce buffer and without faulting a page per
    4 KiB mid-transfer.

    Host-only: no device is involved, so these run without an NPU.
    """

    def test_returns_zeroed_uint8_tensor(self):
        nbytes = 4 * 1024 * 1024
        t = torch.rbln.huge_host_empty(nbytes)
        self.assertEqual(t.device.type, "cpu")
        self.assertEqual(t.dtype, torch.uint8)
        self.assertEqual(t.numel(), nbytes)
        # Zero-filled is part of the contract, though not a discriminating check:
        # freshly mapped pages read as zero anyway, so this catches a wrapper that
        # stopped zero-filling only by luck. The prefaulting it comes from is not
        # observable from here.
        self.assertTrue(bool((t == 0).all()))

    def test_is_huge_page_aligned(self):
        t = torch.rbln.huge_host_empty(4 * 1024 * 1024)
        self.assertEqual(t.data_ptr() % (2 * 1024 * 1024), 0)

    def test_buffer_outlives_its_python_handle(self):
        # The tensor shares the allocation rather than copying it, so nothing but
        # the buffer-protocol reference keeps it alive. If that reference were
        # missing the write below would land in freed memory.
        t = torch.rbln.huge_host_empty(4 * 1024 * 1024)
        gc.collect()
        t[:1024] = 7
        self.assertTrue(bool((t[:1024] == 7).all()))
        self.assertTrue(bool((t[1024:] == 0).all()))

    def test_rejects_zero_size(self):
        with self.assertRaises(ValueError):
            torch.rbln.huge_host_empty(0)

    def test_rejects_size_above_the_supported_bound(self):
        # The bound is ours, and stricter than the fault it exists for: near the top
        # of a uint64 the provider's round-up to the alignment wraps to zero, so it
        # allocates nothing and then prefaults the original size over it.
        for nbytes in (sys.maxsize + 1, (1 << 64) - 1):
            with self.assertRaises(ValueError):
                torch.rbln.huge_host_empty(nbytes)

    def test_rejects_negative_and_non_integer(self):
        with self.assertRaises(ValueError):
            torch.rbln.huge_host_empty(-1)
        with self.assertRaises(TypeError):
            torch.rbln.huge_host_empty(4096.0)


class TestHostRegisterArguments(TestCase):
    """``torch.rbln.host_register`` argument checks; no device needed."""

    def test_rejects_non_positive_and_non_integer(self):
        for ptr, nbytes in ((0, 4096), (-4096, 4096), (4096, 0), (4096, -1)):
            with self.assertRaises(ValueError):
                torch.rbln.host_register(ptr, nbytes)
        with self.assertRaises(TypeError):
            torch.rbln.host_register(4096.0, 4096)
        with self.assertRaises(TypeError):
            torch.rbln.host_unregister(4096.0)

    def test_no_device_leaves_the_range_unregistered(self):
        slab = torch.rbln.huge_host_empty(1 << 20)
        with mock.patch.object(torch_rbln.memory, "_no_rbln_device", return_value=True):
            self.assertFalse(torch.rbln.host_register(slab.data_ptr(), slab.numel()))
            torch.rbln.host_unregister(slab.data_ptr())  # no-op, does not raise


@pytest.mark.npu(count=1)
class TestHostRegister(TestCase):
    """Copies over a registered host range stay correct, including across
    re-registration.
    """

    rbln_device = torch.device("rbln:0")
    nbytes = 8 << 20

    def _round_trip(self, slab, offset, n, fill):
        src = slab[offset : offset + n]
        src.copy_(torch.arange(n, dtype=torch.int64).remainder(251).to(torch.uint8).add_(fill))
        expected = src.clone()
        # Bound so both copies DMA against the slab; a lazy allocation never would.
        on_device = torch.empty(n, dtype=torch.uint8, device=self.rbln_device)
        torch.rbln.bind_device_memory(on_device)
        on_device.copy_(src)
        dst = slab[offset + n : offset + 2 * n]
        dst.zero_()
        dst.copy_(on_device)
        self.assertEqual(dst, expected)

    def test_copies_through_a_registered_range(self):
        slab = torch.rbln.huge_host_empty(self.nbytes)
        self.assertTrue(torch.rbln.host_register(slab.data_ptr(), slab.numel(), self.rbln_device))
        try:
            # Page-aligned (registered path) and unaligned (bounced) slices.
            self._round_trip(slab, 1 << 20, 1 << 20, fill=1)
            self._round_trip(slab, (4 << 20) + 100, 1 << 20, fill=2)
        finally:
            torch.rbln.host_unregister(slab.data_ptr(), self.rbln_device)

    def test_copies_stay_correct_across_unregister_and_register(self):
        # Same host address throughout: a cached command buffer must not reuse a stale
        # device address.
        slab = torch.rbln.huge_host_empty(self.nbytes)
        self.assertTrue(torch.rbln.host_register(slab.data_ptr(), slab.numel(), self.rbln_device))
        self._round_trip(slab, 0, 1 << 20, fill=3)
        torch.rbln.host_unregister(slab.data_ptr(), self.rbln_device)
        self._round_trip(slab, 0, 1 << 20, fill=4)
        self.assertTrue(torch.rbln.host_register(slab.data_ptr(), slab.numel(), self.rbln_device))
        try:
            self._round_trip(slab, 0, 1 << 20, fill=5)
        finally:
            torch.rbln.host_unregister(slab.data_ptr(), self.rbln_device)

    def test_overlapping_range_is_refused(self):
        slab = torch.rbln.huge_host_empty(self.nbytes)
        self.assertTrue(torch.rbln.host_register(slab.data_ptr(), slab.numel(), self.rbln_device))
        try:
            with self.assertRaises(RuntimeError):
                torch.rbln.host_register(slab.data_ptr() + 4096, 4096, self.rbln_device)
        finally:
            torch.rbln.host_unregister(slab.data_ptr(), self.rbln_device)

    def test_unregister_of_an_unregistered_range_raises(self):
        slab = torch.rbln.huge_host_empty(1 << 20)
        with self.assertRaises(RuntimeError):
            torch.rbln.host_unregister(slab.data_ptr(), self.rbln_device)

    def test_unregister_needs_the_start_of_the_range(self):
        slab = torch.rbln.huge_host_empty(1 << 20)
        self.assertTrue(torch.rbln.host_register(slab.data_ptr(), slab.numel(), self.rbln_device))
        try:
            with self.assertRaises(RuntimeError):
                torch.rbln.host_unregister(slab.data_ptr() + 4096, self.rbln_device)
        finally:
            torch.rbln.host_unregister(slab.data_ptr(), self.rbln_device)

    def test_partial_registration_keeps_direct_and_bounced_copies_correct(self):
        slab = torch.rbln.huge_host_empty(self.nbytes)
        with (
            mock.patch.dict(os.environ, TORCH_RBLN_HOST_REGISTRATION_PROFILE="conservative"),
            mock.patch.object(admission, "_TABLE_LIMIT", 1),
        ):
            handle = torch.rbln.host_register_memory(slab, self.rbln_device)
        try:
            self.assertEqual(handle.registered_bytes, 2 << 20)
            self.assertFalse(handle.fully_registered)
            self.assertTrue(handle.covers(slab.data_ptr(), 1 << 20))
            self.assertFalse(handle.covers(slab.data_ptr() + (3 << 20), 1 << 20))
            self._round_trip(slab, 0, 512 << 10, fill=1)
            self._round_trip(slab, 3 << 20, 1 << 20, fill=2)
            self._round_trip(slab, 1 << 20, 768 << 10, fill=3)
        finally:
            handle.close()
        self.assertFalse(torch.rbln.is_host_memory_registered(slab.data_ptr(), 4096, self.rbln_device))

    def test_copy_crossing_two_registered_ranges_is_correct(self):
        slab = torch.rbln.huge_host_empty(self.nbytes)
        with (
            mock.patch.dict(os.environ, TORCH_RBLN_HOST_REGISTRATION_PROFILE="thp"),
            mock.patch.object(admission, "_WINDOW", 2 << 20),
        ):
            handle = torch.rbln.host_register_memory(slab, self.rbln_device)
        try:
            self.assertTrue(handle.fully_registered)
            self.assertEqual(len(handle.registered_ranges), 4)
            self.assertFalse(handle.covers(slab.data_ptr(), self.nbytes))
            self._round_trip(slab, 1 << 20, 768 << 10, fill=4)
        finally:
            handle.close()


class TestHostRegistrationPolicy(TestCase):
    """Admission and ownership without spending real device page tables."""

    def setUp(self):
        super().setUp()

        registration = admission
        self.registration = registration
        self.huge = 2 << 20
        self.ptr = 32 << 20
        self.patches = []
        for name, value in (("_groups", {}), ("_budgets", {}), ("_TABLE_LIMIT", 1), ("_WINDOW", 16 << 20)):
            patch = mock.patch.object(registration, name, value)
            patch.start()
            self.patches.append(patch)
        self.profile = mock.patch.object(registration, "host_registration_profile", return_value="thp")
        self.profile.start()
        self.patches.append(self.profile)
        self.snapshots = []
        self.huge_ranges = []

        def snapshot(_ptr, _size, *, promote):
            self.assertIsInstance(promote, bool)
            value = mock.Mock(huge_ranges=self.huge_ranges, error=0)
            self.snapshots.append(value)
            return value

        for name, kwargs in (
            ("_HostPageSnapshot", {"side_effect": snapshot}),
            ("_host_register", {"return_value": True}),
            ("_host_unregister", {"return_value": None}),
        ):
            patch = mock.patch.object(torch_rbln._C, name, **kwargs)
            replacement = patch.start()
            setattr(self, name, replacement)
            self.patches.append(patch)

    def tearDown(self):
        self._host_unregister.side_effect = None
        for state in list(self.registration._groups.values()):
            state.close()
        for patch in reversed(self.patches):
            patch.stop()
        super().tearDown()

    def managed(self, size, ptr=None, device=0, owner=None):
        return self.registration.register_managed(self.ptr if ptr is None else ptr, size, device, owner, available=True)

    def test_partial_registration_keeps_huge_regions_after_budget_exhaustion(self):
        h, p = self.huge, self.ptr
        self.huge_ranges = [(p, h), (p + 2 * h, h), (p + 4 * h, h)]
        handle = self.managed(5 * h)
        self.assertEqual(handle.registered_ranges, ((p, 3 * h), (p + 4 * h, h)))
        self.assertEqual(handle.registered_bytes, 4 * h)
        self.assertFalse(handle.fully_registered)
        self.assertTrue(handle.covers(p + h, h))
        self.assertFalse(handle.covers(p + 2 * h, 3 * h))
        self.assertEqual(self.registration._budgets[0].tables, 1)
        self.assertEqual(self._host_register.call_count, 2)
        handle.close()
        self.assertEqual(self.registration._budgets[0].tables, 0)

    def test_strict_registration_rolls_back_prior_windows(self):
        h, p = self.huge, self.ptr
        self.huge_ranges = [(p, 2 * h)]
        with mock.patch.object(self.registration, "_WINDOW", 2 * h):
            self.assertFalse(self.registration.register_strict(p, 4 * h, 0))
        self._host_register.assert_called_once_with(0, p, 2 * h)
        self._host_unregister.assert_called_once_with(0, p)
        self.assertEqual(self.registration._budgets[0].tables, 0)

    def test_temporary_pins_survive_native_registration(self):
        def register(*_args):
            self.snapshots[-1].close.assert_not_called()
            return True

        self._host_register.side_effect = register
        handle = self.managed(self.huge)
        self.snapshots[0].close.assert_called_once()
        handle.close()

    def test_unavailable_snapshot_charges_the_entire_range(self):
        self._HostPageSnapshot.side_effect = lambda *_args, **_kwargs: mock.Mock(huge_ranges=[], error=1)
        handle = self.managed(3 * self.huge)
        self.assertEqual(handle.registered_bytes, self.huge)
        self._host_register.assert_called_once_with(0, self.ptr, self.huge)
        handle.close()

    def test_unaligned_head_does_not_destroy_body_huge_alignment(self):
        h, p = self.huge, self.ptr + 4096
        self.huge_ranges = [(self.ptr + h, 2 * h)]
        with mock.patch.object(self.registration, "_TABLE_LIMIT", 2):
            handle = self.managed(3 * h, ptr=p)
        self.assertEqual(handle.registered_ranges, ((p, h - 4096), (self.ptr + h, 2 * h + 4096)))
        self.assertEqual(self.registration._budgets[0].tables, 2)
        handle.close()

    def test_small_registration_consumes_a_whole_table(self):
        first = self.managed(4096)
        second = self.managed(4096, ptr=self.ptr + self.huge)
        self.assertTrue(first.fully_registered)
        self.assertEqual(second.registered_bytes, 0)
        self.assertEqual(self._host_register.call_count, 1)
        first.close()
        second.close()

    def test_all_huge_registration_still_obeys_total_va_limit(self):
        self.huge_ranges = [(self.ptr, 3 * self.huge)]
        with mock.patch.object(self.registration, "_VA_LIMIT", 2 * self.huge):
            handle = self.managed(3 * self.huge)
        self.assertEqual(handle.registered_bytes, 2 * self.huge)
        self.assertEqual(self.registration._budgets[0].tables, 0)
        handle.close()

    def test_adjacent_registrations_do_not_imply_one_direct_transfer(self):
        self.huge_ranges = [(self.ptr, 2 * self.huge)]
        with mock.patch.object(self.registration, "_WINDOW", self.huge):
            handle = self.managed(2 * self.huge)
        self.assertTrue(handle.fully_registered)
        self.assertFalse(handle.covers(self.ptr, 2 * self.huge))
        self.assertTrue(handle.covers(self.ptr + self.huge, self.huge))
        with self.assertRaises(TypeError):
            bool(handle)
        handle.close()

    def test_strict_and_partial_apis_share_the_same_budget(self):
        self.assertTrue(self.registration.register_strict(self.ptr, self.huge, 0))
        partial = self.managed(self.huge, ptr=self.ptr + self.huge)
        self.assertEqual(partial.registered_bytes, 0)
        self.registration.unregister_strict(self.ptr, 0)
        partial.close()
        admitted = self.managed(self.huge, ptr=self.ptr + self.huge)
        self.assertTrue(admitted.fully_registered)
        admitted.close()

    def test_failed_unpin_retains_owner_budget_and_disables_coverage(self):
        owner = torch.empty(4)
        ref = weakref.ref(owner)
        handle = self.managed(self.huge, owner=owner)
        self._host_unregister.side_effect = RuntimeError("unpin failed after registry erase")
        with self.assertRaisesRegex(RuntimeError, "registry erase"):
            handle.close()
        del owner
        gc.collect()
        self.assertIsNotNone(ref())
        self.assertFalse(handle.covers(self.ptr, self.huge))
        self.assertEqual(self.registration._budgets[0].tables, 1)
        self._host_unregister.side_effect = None
        handle.close()
        gc.collect()
        self.assertIsNone(ref())

    def test_ambiguous_register_failure_is_not_reported_as_unregistered(self):
        self._host_register.side_effect = RuntimeError("register failed")
        self._host_unregister.side_effect = RuntimeError("release not confirmed")
        with self.assertRaisesRegex(RuntimeError, "register failed"):
            self.managed(self.huge)
        self.assertEqual(self.registration._budgets[0].tables, 1)
        self.assertFalse(self.registration.is_registered(self.ptr, self.huge, 0))

    def test_unsupported_runtime_leaves_no_registration_or_charge(self):
        self._host_register.return_value = False
        handle = self.managed(self.huge)
        self.assertEqual(handle.registered_bytes, 0)
        self.assertEqual(self.registration._budgets[0].tables, 0)
        self._host_unregister.assert_not_called()
        handle.close()

    def test_concurrent_requests_cannot_spend_the_same_allowance(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            handles = list(pool.map(lambda index: self.managed(self.huge, ptr=self.ptr + index * self.huge), range(4)))
        self.assertEqual(sum(h.registered_bytes for h in handles), self.huge)
        self._host_register.assert_called_once()
        for handle in handles:
            handle.close()

    def test_failed_registration_closes_snapshot_even_on_error(self):
        self._host_register.side_effect = RuntimeError("register failed")
        with self.assertRaisesRegex(RuntimeError, "register failed"):
            self.managed(self.huge)
        self.snapshots[0].close.assert_called_once()
        self.assertEqual(self.registration._budgets[0].tables, 0)

    def test_device_budgets_are_independent(self):
        first, second = self.managed(self.huge, device=0), self.managed(self.huge, device=1)
        self.assertTrue(first.fully_registered)
        self.assertTrue(second.fully_registered)
        first.close()
        second.close()

    def test_default_profile_never_credits_unverified_huge_pages(self):
        with mock.patch.object(self.registration, "host_registration_profile", return_value="conservative"):
            self.huge_ranges = [(self.ptr, 2 * self.huge)]
            handle = self.managed(2 * self.huge)
        self.assertEqual(handle.registered_bytes, self.huge)
        self._HostPageSnapshot.assert_not_called()
        handle.close()

    def test_forked_child_cannot_spend_or_release_parent_registrations(self):
        handle = self.managed(self.huge)
        with (
            mock.patch.object(self.registration, "_forked_with_registrations", new=True),
            self.assertRaisesRegex(RuntimeError, "spawn"),
        ):
            self.managed(self.huge, ptr=self.ptr + self.huge)
        with mock.patch.object(os, "getpid", return_value=os.getpid() + 1):
            self.assertFalse(handle.covers(self.ptr, self.huge))
            self.assertFalse(handle.fully_registered)
            self.assertEqual(handle.registered_ranges, ())
            with self.assertRaisesRegex(RuntimeError, "forked child"):
                handle.close()
        handle.close()

    def test_handle_creation_failure_rolls_back_registration(self):
        with (
            mock.patch.object(self.registration, "HostMemoryRegistration", side_effect=MemoryError),
            self.assertRaises(MemoryError),
        ):
            self.managed(self.huge)
        self._host_unregister.assert_called_once_with(0, self.ptr)
        self.assertEqual(self.registration._budgets[0].tables, 0)

    def test_partial_handle_is_not_released_through_the_strict_api(self):
        handle = self.managed(self.huge)
        with self.assertRaisesRegex(RuntimeError, "close"):
            self.registration.unregister_strict(self.ptr, 0)
        self._host_unregister.assert_not_called()
        handle.close()

    def test_overlap_in_unregistered_part_is_rejected(self):
        handle = self.managed(2 * self.huge)
        with self.assertRaisesRegex(RuntimeError, "overlaps"):
            self.managed(self.huge, ptr=self.ptr + self.huge)
        handle.close()

    def test_finalizer_releases_registration_before_owner(self):
        owner = torch.empty(4)
        ref = weakref.ref(owner)
        handle = self.managed(self.huge, owner=owner)
        del owner
        self.assertIsNotNone(ref())
        del handle
        gc.collect()
        self._host_unregister.assert_called_once_with(0, self.ptr)
        self.assertIsNone(ref())
        self.assertEqual(self.registration._budgets[0].tables, 0)

    def test_failed_finalizer_retains_owner_and_budget(self):
        owner = torch.empty(4)
        ref = weakref.ref(owner)
        handle = self.managed(self.huge, owner=owner)
        self._host_unregister.side_effect = RuntimeError("release not confirmed")
        del handle, owner
        gc.collect()
        self.assertIsNotNone(ref())
        self.assertEqual(self.registration._budgets[0].tables, 1)
        self.assertFalse(self.registration.is_registered(self.ptr, self.huge, 0))
        self._host_unregister.side_effect = None
        self.registration._groups[0, self.ptr].close()
        gc.collect()
        self.assertIsNone(ref())

    def test_partial_close_keeps_only_failed_entry_charged(self):
        h, p = self.huge, self.ptr
        self.huge_ranges = [(p, 3 * h)]
        with mock.patch.object(self.registration, "_WINDOW", h):
            handle = self.managed(3 * h)
        self._host_unregister.side_effect = [None, RuntimeError("unpin failed"), None]
        with self.assertRaisesRegex(RuntimeError, "unpin failed"):
            handle.close()
        self.assertEqual(self._host_unregister.call_count, 3)
        self.assertEqual(self.registration._budgets[0].va_bytes, h)
        self.assertEqual(handle.registered_ranges, ())
        self._host_unregister.side_effect = None
        handle.close()
        self.assertEqual(self.registration._budgets[0].va_bytes, 0)
        self._host_unregister.assert_called_with(0, p + h)


class TestHostPageSnapshot(TestCase):
    """Exercise real Linux page inspection without opening an NPU."""

    def test_invalid_ranges(self):
        for ptr, size in ((0, 4096), (4096, 0), (4096, (1 << 30) + 1), ((1 << 64) - 4096, 4096)):
            with self.assertRaises(ValueError):
                torch_rbln._C._HostPageSnapshot(ptr, size, promote=False)

    def test_real_anonymous_mixed_pages(self):
        h = 2 << 20
        buffer = torch.rbln.huge_host_empty(4 * h)
        ptr = buffer.data_ptr()
        libc = ctypes.CDLL(None, use_errno=True)
        libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        libc.madvise.restype = ctypes.c_int
        for address, size, advice in (
            (ptr, 2 * h, mmap.MADV_HUGEPAGE),
            (ptr, 2 * h, 25),  # MADV_COLLAPSE; best effort on older kernels.
            (ptr + 2 * h, 2 * h, mmap.MADV_NOHUGEPAGE),
            (ptr + 2 * h, 2 * h, mmap.MADV_DONTNEED),
        ):
            result = libc.madvise(address, size, advice)
            if advice != 25:
                self.assertEqual(result, 0)
        buffer[2 * h :].zero_()
        snapshot = torch_rbln._C._HostPageSnapshot(ptr, buffer.numel(), promote=False)
        try:
            if snapshot.error:
                self.skipTest(f"io_uring/PAGEMAP_SCAN unavailable: errno={snapshot.error}")
            self.assertTrue(all(start + size <= ptr + 2 * h for start, size in snapshot.huge_ranges))
            if not snapshot.huge_ranges:
                self.skipTest("THP unavailable for the huge half of this fixture")
            self.assertEqual(snapshot.huge_ranges, [(ptr, 2 * h)])
            # The temporary long-term pin must retain that physical huge page
            # even if the process stops asking for future THP allocations.
            self.assertEqual(libc.madvise(ptr, 2 * h, mmap.MADV_NOHUGEPAGE), 0)
            second = torch_rbln._C._HostPageSnapshot(ptr, 2 * h, promote=False)
            try:
                self.assertEqual(second.error, 0)
                self.assertEqual(second.huge_ranges, [(ptr, 2 * h)])
            finally:
                second.close()
        finally:
            snapshot.close()
        self.assertEqual(snapshot.huge_ranges, [])

    def test_real_posix_shared_memory_base_pages(self):
        shared = SharedMemory(create=True, size=4 << 20)
        buffer = torch.frombuffer(shared.buf, dtype=torch.uint8)
        ptr = buffer.data_ptr()
        libc = ctypes.CDLL(None, use_errno=True)
        libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        libc.madvise.restype = ctypes.c_int
        try:
            self.assertEqual(libc.madvise(ptr, buffer.numel(), mmap.MADV_NOHUGEPAGE), 0)
            buffer.zero_()
            snapshot = torch_rbln._C._HostPageSnapshot(ptr, buffer.numel(), promote=False)
            try:
                if snapshot.error:
                    self.skipTest(f"io_uring/PAGEMAP_SCAN unavailable: errno={snapshot.error}")
                self.assertEqual(snapshot.huge_ranges, [])
            finally:
                snapshot.close()
        finally:
            del buffer
            shared.close()
            shared.unlink()


instantiate_device_type_tests(TestAliasedTensors, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestInputOutputTensors, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestBindDeviceMemory, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
