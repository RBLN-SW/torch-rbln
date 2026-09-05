# Owner(s): ["module: PrivateUse1"]

"""
Test suite for tensor memory correctness.

Validates that device operations produce correct results under non-trivial memory scenarios:
- Storage aliasing: multiple tensors sharing the same underlying storage
- Input/output memory independence: ensuring that input and output tensors do not share overlapping memory
- Bound allocations: device buffers handed to a consumer that DMAs out of band
- Huge host buffers: the host side of the same transfers
"""

import gc
import os
import sys
from unittest import mock

import pytest
import torch
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase

import torch_rbln._C
from test.utils import run_in_isolated_process, SUPPORTED_DTYPES


# Tolerance for numerical comparisons
ATOL = 0.01
RTOL = 0.01


@pytest.mark.test_set_ci
class TestAliasedTensors(TestCase):
    binary_ops = [torch.add]
    rbln_device = torch.device("rbln:0")
    shapes = [(2, 16), (2, 64)]

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
    """Worker function that runs in a spawned subprocess to ensure a clean data_ptr counter."""
    # Create a seed input tensor whose internal key value may collide with output tensor memory keys.
    seed_tensor = torch.randn([2, 2], dtype=dtype, device=rbln_device)
    seed_data_ptr = seed_tensor.data_ptr()

    rbln_out = torch.abs(seed_tensor)  # First output tensor allocation.
    cpu_out = torch.abs(seed_tensor.cpu())
    torch.testing.assert_close(rbln_out.cpu(), cpu_out, atol=ATOL, rtol=RTOL)

    # Burn through (seed_data_ptr - 1) output tensor allocations so the next output tensor receives a key equal to
    # seed_data_ptr, forcing a key collision.
    for i in range(seed_data_ptr - 1):
        # Vary the size of the tensor to ensure unique memory allocation.
        t = torch.randn([i], dtype=dtype, device=rbln_device)
        rbln_out = torch.abs(t)
        cpu_out = torch.abs(t.cpu())
        torch.testing.assert_close(rbln_out.cpu(), cpu_out, atol=ATOL, rtol=RTOL)
    # After the loop, the next output tensor key may collide with the seed tensor key.

    x = torch.randn([2, 2, 4], dtype=dtype, device=rbln_device)
    y = torch.randn([2, 2, 4], dtype=dtype, device=rbln_device)

    # Use trunc-mode division to amplify numerical divergence if the backend silently returns wrong data.
    rbln_out = torch.div(x, y, rounding_mode="trunc")
    cpu_out = torch.div(x.cpu(), y.cpu(), rounding_mode="trunc")
    torch.testing.assert_close(rbln_out.cpu(), cpu_out, atol=ATOL, rtol=RTOL)


@pytest.mark.test_set_ci
class TestInputOutputTensors(TestCase):
    """
    Regression tests for input and output tensor memory collisions.

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
    """Assert ``bind_device_memory`` raises rather than faulting once the runtime is down.

    Runs in a spawned process: the shutdown flag is process-global and set-once, so
    flipping it here would poison every later test on this worker.
    """
    # Third Party
    import torch

    import torch_rbln._C

    t = torch.empty(4096, dtype=torch.uint8, device=device)
    torch_rbln._C._set_runtime_shutting_down(True)
    try:
        torch.rbln.bind_device_memory(t)
    except RuntimeError:
        return
    raise AssertionError("bind_device_memory did not raise with the runtime shutting down")


@pytest.mark.test_set_ci
class TestBindDeviceMemory(TestCase):
    """
    ``torch.rbln.bind_device_memory`` materializes a tensor's device allocation, for
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
        # The runtime may bring more along -- a slab it grew to serve this -- but not less.
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

    def test_chiplet_count_is_positive(self):
        # Every device has at least one chiplet; a device with chiplet-partitioned DRAM
        # reports how many pools it has. The count is the range ``chiplet=`` accepts.
        self.assertGreaterEqual(torch.rbln.chiplet_count(self.rbln_device), 1)

    def test_bind_on_a_chiplet_lands_there(self):
        # The placement is what the per-chiplet statistics make observable: binding on
        # chiplet k grows that chiplet's allocated bytes by the allocation and no other's.
        n = torch.rbln.chiplet_count(self.rbln_device)
        if n < 2:
            self.skipTest("placement is only observable with more than one chiplet")
        nbytes = 8 << 20
        chiplet = n - 1
        with mock.patch.dict(os.environ):
            os.environ.pop("TORCH_RBLN_EAGER_MALLOC", None)  # would bind at allocation
            torch.rbln.synchronize()
            t = torch.empty(nbytes, dtype=torch.uint8, device=self.rbln_device)
            before = torch.rbln.memory_stats_per_chiplet(self.rbln_device)
            torch.rbln.bind_device_memory(t, chiplet=chiplet)
            after = torch.rbln.memory_stats_per_chiplet(self.rbln_device)
        self.assertEqual(len(before), n)
        self.assertEqual(len(after), n)
        key = "allocated.current"
        grown = [after[i][key] - before[i][key] for i in range(n)]
        self.assertGreaterEqual(grown[chiplet], nbytes)
        for i in range(n):
            if i != chiplet:
                self.assertEqual(grown[i], 0, f"chiplet {i} grew by {grown[i]} bytes")

    def test_placement_survives_a_compiled_op(self):
        # A compiled program names chiplet 0 for its operands; binding a tensor placed
        # elsewhere as one of them must not move it. Observable as the address staying put.
        n = torch.rbln.chiplet_count(self.rbln_device)
        if n < 2:
            self.skipTest("placement is only observable with more than one chiplet")
        # 32 MiB bf16 so the permuted copy takes the compiled path rather than the
        # strided walk (see RBLNCopy.cpp: only bf16 is bit-exact there).
        src = torch.empty((16, 8, 1024, 128), dtype=torch.bfloat16, device=self.rbln_device)
        out = torch.empty((16, 1024, 8, 128), dtype=torch.bfloat16, device=self.rbln_device)
        torch.rbln.bind_device_memory(src, chiplet=n - 1)
        torch.rbln.bind_device_memory(out, chiplet=n - 1)
        before = torch.rbln.memory_stats_per_chiplet(self.rbln_device)
        out.copy_(src.permute(0, 2, 1, 3))
        torch.rbln.synchronize()
        after = torch.rbln.memory_stats_per_chiplet(self.rbln_device)
        key = "allocated.current"
        # The program brings its own buffers to chiplet 0 -- an output-sized DramTensor it
        # allocates at load whether or not the caller supplies the output, plus command
        # streams -- so chiplet 0 does grow by about one operand. Migrating even one
        # operand there would add another operand on top of that.
        operand_bytes = src.numel() * src.element_size()
        self.assertLess(after[0][key] - before[0][key], 2 * operand_bytes)
        # ...and the operands did not leave their chiplet.
        self.assertGreaterEqual(after[n - 1][key], before[n - 1][key])

    def test_rejects_out_of_range_chiplet(self):
        t = torch.empty(4096, dtype=torch.uint8, device=self.rbln_device)
        with self.assertRaises(RuntimeError):
            torch.rbln.bind_device_memory(t, chiplet=torch.rbln.chiplet_count(self.rbln_device))
        with self.assertRaises(ValueError):
            torch.rbln.bind_device_memory(t, chiplet=-1)

    def test_rejects_interior_view(self):
        # A slice's data_ptr() is an interior address; binding it would configure the
        # wrong region, so it is refused rather than silently mis-bound. A view that
        # still spans the whole storage carries the allocation's own address and size,
        # and is accepted.
        base = torch.empty(4096, dtype=torch.uint8, device=self.rbln_device)
        with self.assertRaises(RuntimeError):
            torch.rbln.bind_device_memory(base[16:])
        torch.rbln.bind_device_memory(base.view(64, 64))

    @pytest.mark.single_worker
    def test_raises_once_runtime_is_shutting_down(self):
        # The raw runtime call would fault rather than raise past teardown, and this
        # entry point is public, so a caller can reach it there.
        run_in_isolated_process(_bind_after_shutdown_worker, str(self.rbln_device))


@pytest.mark.test_set_ci
class TestHugeHostEmpty(TestCase):
    """
    ``torch.rbln.huge_host_empty`` returns host memory the device can DMA into
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


instantiate_device_type_tests(TestAliasedTensors, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestInputOutputTensors, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestBindDeviceMemory, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
