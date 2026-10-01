# Owner(s): ["module: PrivateUse1"]

"""
End-to-end test for the copies in ``cpu_fallback_rbln`` (see
``aten/src/ATen/native/rbln/RBLNCPUFallback.cpp``).

When an op falls through to the CPU kernel, its rbln-device inputs are copied
into CPU tensors (a pure ``out=`` gets fresh CPU storage instead), the CPU op
runs, and every write-aliasing input is copied back to the device, resized if
the kernel resized it. These tests verify representative fallback ops produce
correct results and the ``out=`` resize contract holds.
"""

import warnings

import pytest
import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


@pytest.mark.test_set_ci
class TestCPUFallbackCopy(TestCase):
    """Ops routed to the CPU fallback must yield correct results. ``aten::sigmoid``
    on int32 is intentionally a non-fp16 input (forces the dispatch shim's pre-check
    shortcut into the C++ fallback), and is a unary op with no write aliasing."""

    def test_fallback_unary_int32_input(self) -> None:
        x = torch.arange(48, dtype=torch.int32, device="rbln").reshape(6, 8)
        out = torch.sigmoid(x.float()).to("cpu")
        # Reference computation on CPU.
        ref = torch.sigmoid(torch.arange(48, dtype=torch.int32).reshape(6, 8).float())
        self.assertEqual(out, ref)

    def test_fallback_returns_device_tensor(self) -> None:
        """Sanity: a rbln tensor flowing through cpu_fallback_rbln does not
        crash and returns a tensor on the rbln device with the expected
        shape."""
        x = torch.arange(16, dtype=torch.int32, device="rbln")
        # int32 input forces the cpu_fallback_rbln path (fp16-only kernels).
        y = torch.sigmoid(x.float())
        self.assertEqual(y.device.type, "rbln")
        self.assertEqual(tuple(y.shape), (16,))

    def test_pure_out_undersized_grows(self) -> None:
        """Undersized pure ``out=`` must GROW to the result shape, and the "output was
        resized" warning fires once. ``sin`` is a CPU-fallback op."""
        x = torch.arange(8, dtype=torch.float32, device="rbln")
        out = torch.empty(1, dtype=torch.float32, device="rbln")  # undersized -> grow
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ret = torch.sin(x, out=out)
        resized = [w for w in caught if issubclass(w.category, UserWarning) and "was resized" in str(w.message)]
        self.assertEqual(len(resized), 1, f"resize warning should fire once, got {len(resized)}")
        self.assertEqual(tuple(ret.shape), (8,))
        self.assertTrue(ret is out)
        self.assertEqual(ret.to("cpu"), torch.sin(torch.arange(8, dtype=torch.float32)))

    def test_pure_out_oversized_shrinks(self) -> None:
        """The shrink counterpart: an oversized ``out=`` comes back with the
        result's shape."""
        x = torch.arange(8, dtype=torch.float32, device="rbln")
        out = torch.empty(16, dtype=torch.float32, device="rbln")  # oversized -> shrink
        with pytest.warns(UserWarning):
            ret = torch.sin(x, out=out)
        self.assertEqual(tuple(ret.shape), (8,))
        self.assertEqual(ret.to("cpu"), torch.sin(torch.arange(8, dtype=torch.float32)))

    def test_zero_k_matmul_into_filled_out(self) -> None:
        """A 0-K matmul yields an all-zero (5, 10) output. Written into an ``out=``
        tensor filled with NaN, it must replace every element; comparing two rbln
        tensors routes ``eq`` through the fallback, so this checks the copy back
        and the fallback's input copies at once."""
        a = torch.zeros(5, 0, dtype=torch.float16, device="rbln")
        b = torch.zeros(0, 10, dtype=torch.float16, device="rbln")
        out = torch.full((5, 10), float("nan"), dtype=torch.float16, device="rbln")
        torch.mm(a, b, out=out)
        # rbln-vs-rbln compare -> isclose/eq -> cpu_fallback copies `out` to the host.
        self.assertEqual(out, torch.zeros(5, 10, dtype=torch.float16, device="rbln"))


instantiate_device_type_tests(TestCPUFallbackCopy, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
