# Owner(s): ["module: PrivateUse1"]

"""
Test suite for SDPA decode-phase overflow behavior.
"""

import math

import pytest
import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

from torch_rbln._internal.kernels.sdpa import (
    _sdpa_cpu_fallback,
    can_use_rbln_sdpa,
    needs_sdpa_shape_fallback,
    scaled_dot_product_fused_attention_overrideable_rbln,
)


@pytest.mark.test_set_ci
class TestSDPADecodeOverflow(TestCase):
    """Test SDPA decode-phase overflow behavior."""

    def _make_qwen_like_attention_tensors(
        self,
        batch_size: int = 1,
        num_heads: int = 12,
        seq_len_q: int = 1,
        seq_len_kv: int = 128,
        head_dim: int = 128,
        device: str = "rbln",
        dtype: torch.dtype = torch.float16,
        overflow_heads: list[int] | None = None,
    ):
        """Create attention tensors mimicking Qwen2.5's behavior.

        Some heads have K/V values that exceed float16 safe range (~65504),
        which causes overflow when computing Q @ K^T in float16 on CPU.
        To handle this, the CPU fallback should upcast to float32 internally.

        Args:
            overflow_heads: list of head indices to set with overflow-prone values.
                If None, defaults to [0, 1] to simulate Qwen2.5 behavior.
        """
        if overflow_heads is None:
            overflow_heads = [0, 1]

        # Create normal Q, K, V tensors
        query = torch.randn(batch_size, num_heads, seq_len_q, head_dim, dtype=dtype, device="cpu")
        key = torch.randn(batch_size, num_heads, seq_len_kv, head_dim, dtype=dtype, device="cpu")
        value = torch.randn(batch_size, num_heads, seq_len_kv, head_dim, dtype=dtype, device="cpu")

        # For overflow heads: scale K values to make Q @ K^T overflow in float16
        # In Qwen2.5, certain heads have learned large key values
        for h in overflow_heads:
            # Scale key values so that dot product (sum of head_dim products) overflows float16
            # float16 max is ~65504; with head_dim=128, each product needs ~sqrt(65504/128) ≈ ~22.6
            # Use values around 30-50 to reliably trigger overflow
            key[:, h, :, :] = key[:, h, :, :] * 40.0

        query = query.to(device)
        key = key.to(device)
        value = value.to(device)

        return query, key, value

    def test_decode_phase_shape_fallback_detected(self):
        """Verify that decode-phase tensors (seq_len=1) trigger shape fallback."""
        query = torch.randn(1, 12, 1, 128, dtype=torch.float16, device="rbln")
        key = torch.randn(1, 12, 128, 128, dtype=torch.float16, device="rbln")

        needs_fallback, reason = needs_sdpa_shape_fallback(query, key)
        self.assertTrue(needs_fallback, "Decode-phase seq_len=1 should trigger shape fallback")
        self.assertIn("not aligned", reason)

    def test_decode_phase_cant_use_rbln(self):
        """Verify that decode-phase tensors are rejected by can_use_rbln_sdpa."""
        query = torch.randn(1, 12, 1, 128, dtype=torch.float16, device="rbln")
        key = torch.randn(1, 12, 128, 128, dtype=torch.float16, device="rbln")
        value = torch.randn(1, 12, 128, 128, dtype=torch.float16, device="rbln")

        can_use, reason = can_use_rbln_sdpa(query, key, value)
        self.assertFalse(can_use, "Decode-phase seq_len=1 should not use RBLN SDPA without padding")

    def test_cpu_fallback_float16_overflow(self):
        """Test that CPU fallback produces valid (non-nan/inf) output even with overflow-prone values.

        This is the core issue: Qwen2.5-like models have heads where K values are large enough
        that Q @ K^T overflows float16 on CPU. Before the fix, CPU fallback would produce
        inf/nan → zero output. After the fix, the fallback should upcast to float32.
        """
        query, key, value = self._make_qwen_like_attention_tensors(
            seq_len_q=1,  # decode phase
            seq_len_kv=128,
            overflow_heads=[0, 1, 2],
        )

        # The CPU fallback should produce valid output (no inf/nan)
        output = _sdpa_cpu_fallback(query, key, value, is_causal=False)
        output_cpu = output.cpu()

        self.assertFalse(
            torch.isnan(output_cpu).any().item(),
            "SDPA CPU fallback should not produce NaN (upcast to float32 should handle overflow)",
        )
        self.assertFalse(
            torch.isinf(output_cpu).any().item(),
            "SDPA CPU fallback should not produce Inf (upcast to float32 should handle overflow)",
        )

        # Output should not be all zeros (the symptom of the overflow issue)
        self.assertFalse(
            (output_cpu == 0).all().item(),
            "SDPA CPU fallback should not produce all-zero output (overflow symptom)",
        )

    def test_cpu_fallback_float16_overflow_with_causal(self):
        """Test CPU fallback with causal mask and overflow-prone values."""
        query, key, value = self._make_qwen_like_attention_tensors(
            seq_len_q=1,
            seq_len_kv=128,
            overflow_heads=[0, 1],
        )

        output = _sdpa_cpu_fallback(query, key, value, is_causal=True)
        output_cpu = output.cpu()

        self.assertFalse(
            torch.isnan(output_cpu).any().item(),
            "SDPA CPU fallback with causal mask should not produce NaN",
        )
        self.assertFalse(
            torch.isinf(output_cpu).any().item(),
            "SDPA CPU fallback with causal mask should not produce Inf",
        )

    def test_overrideable_sdpa_decode_phase(self):
        """Test the full overrideable SDPA path for decode-phase with overflow-prone values.

        This tests the actual entry point that PyTorch calls for SDPA.
        """
        query, key, value = self._make_qwen_like_attention_tensors(
            seq_len_q=1,
            seq_len_kv=128,
            overflow_heads=[0, 1, 2],
        )

        result = scaled_dot_product_fused_attention_overrideable_rbln(
            query,
            key,
            value,
            attn_bias=None,
            dropout_p=0.0,
            is_causal=False,
            scale=None,
        )
        output = result[0]  # first element is the output tensor
        output_cpu = output.cpu()

        self.assertFalse(
            torch.isnan(output_cpu).any().item(),
            "Full SDPA overrideable path should not produce NaN for decode phase",
        )
        self.assertFalse(
            torch.isinf(output_cpu).any().item(),
            "Full SDPA overrideable path should not produce Inf for decode phase",
        )
        self.assertFalse(
            (output_cpu == 0).all().item(),
            "Full SDPA overrideable path should not produce all-zero output",
        )

    def test_cpu_fallback_preserves_device(self):
        """Test that CPU fallback result is on the same device as input."""
        query, key, value = self._make_qwen_like_attention_tensors(
            seq_len_q=1,
            seq_len_kv=128,
        )

        output = _sdpa_cpu_fallback(query, key, value)
        self.assertEqual(
            output.device.type,
            "rbln",
            "CPU fallback result should be on RBLN device",
        )

    def test_cpu_fallback_preserves_dtype(self):
        """Test that CPU fallback result has the same dtype as input."""
        query, key, value = self._make_qwen_like_attention_tensors(
            seq_len_q=1,
            seq_len_kv=128,
        )

        output = _sdpa_cpu_fallback(query, key, value)
        self.assertEqual(
            output.dtype,
            torch.float16,
            "CPU fallback result should preserve float16 dtype",
        )

    def test_cpu_fallback_correctness_vs_float32_ref(self):
        """Test that CPU fallback with upcast produces results close to float32 reference.

        The float32 reference is the "ground truth" since it has enough precision
        to handle the values correctly.
        """
        query, key, value = self._make_qwen_like_attention_tensors(
            seq_len_q=1,
            seq_len_kv=128,
            overflow_heads=[0, 1],
        )

        # Get result from CPU fallback (should upcast internally)
        output = _sdpa_cpu_fallback(query, key, value, is_causal=False)
        output_cpu = output.cpu().float()

        # Compute float32 reference directly on CPU
        q_cpu = query.cpu().float()
        k_cpu = key.cpu().float()
        v_cpu = value.cpu().float()

        scale = 1.0 / math.sqrt(q_cpu.size(-1))
        attn_weights = torch.matmul(q_cpu, k_cpu.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        ref_output = torch.matmul(attn_weights, v_cpu)

        # Allow some tolerance for float16 → float32 → float16 round-trip
        torch.testing.assert_close(
            output_cpu,
            ref_output,
            atol=0.1,
            rtol=0.05,
            msg="CPU fallback output should be close to float32 reference",
        )

    def test_normal_prefill_not_affected(self):
        """Test that aligned prefill shapes are not affected by the fix."""
        # Aligned prefill: seq_len=128 (aligned to 32)
        query = torch.randn(1, 12, 128, 128, dtype=torch.float16, device="rbln")
        key = torch.randn(1, 12, 128, 128, dtype=torch.float16, device="rbln")

        needs_fallback, reason = needs_sdpa_shape_fallback(query, key)
        self.assertFalse(needs_fallback, "Aligned prefill shapes should not trigger fallback")


instantiate_device_type_tests(TestSDPADecodeOverflow, globals(), only_for="privateuse1")

if __name__ == "__main__":
    run_tests()
