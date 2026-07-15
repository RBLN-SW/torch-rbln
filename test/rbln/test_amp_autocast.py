# Owner(s): ["module: PrivateUse1"]

"""AMP / autocast behavior for the RBLN backend.

RBLN does not implement AMP autocast yet (no ``AutocastPrivateUse1`` cast policy
is registered). The dtype catalog therefore advertises an *empty* AMP set, so
entering ``torch.autocast("rbln", ...)`` makes torch disable autocast with a
warning instead of dispatching an op to a missing Autocast kernel (which used to
raise ``NotImplementedError`` on the first op).

Two invariants this suite locks down:
1. The empty catalog is coherent across the C++/Python chain
   (``c10::rbln::kAmpDtypes`` -> ``_amp_dtypes()`` -> ``SupportedDtypes.amp`` ->
   ``get_amp_supported_dtype()``).
2. ``get_amp_supported_dtype`` must still *exist* on the device module:
   transformers/accelerate enter ``torch.autocast(device_type="rbln", ...)``
   (notably the pervasive ``enabled=False`` "force float32" blocks), which
   asserts the function is registered.
"""

import warnings

import pytest
import torch
import torch_rbln  # noqa: F401  (registers the rbln device module + autocast plumbing)
from torch.testing._internal.common_utils import run_tests, TestCase

from torch_rbln._internal.ops_utils import SupportedDtypes


@pytest.mark.test_set_ci
class TestAmpCatalog(TestCase):
    """The AMP dtype catalog is empty and coherent end-to-end."""

    def test_amp_catalog_is_empty(self):
        self.assertIsInstance(SupportedDtypes.amp, tuple)
        self.assertEqual(SupportedDtypes.amp, ())

    def test_get_amp_supported_dtype_empty(self):
        self.assertEqual(torch.rbln.get_amp_supported_dtype(), [])

    def test_get_amp_supported_dtype_still_exists(self):
        # Removing this function makes torch.autocast(device_type="rbln", ...) raise
        # AssertionError, breaking transformers/accelerate. Guard against that.
        self.assertTrue(hasattr(torch.rbln, "get_amp_supported_dtype"))


@pytest.mark.test_set_ci
class TestAutocastDisabledGracefully(TestCase):
    """Entering an rbln autocast region must never crash on the first op."""

    @staticmethod
    def _matmul_rbln():
        a = torch.randn(32, 32, device="rbln")
        b = torch.randn(32, 32, device="rbln")
        return a @ b

    def test_fp16_autocast_disables_with_warning(self):
        with self.assertWarns(UserWarning):
            with torch.autocast(device_type="rbln", dtype=torch.float16):
                y = self._matmul_rbln()
        # Disabled -> the op ran in its native dtype (fp32), not fp16.
        self.assertEqual(y.dtype, torch.float32)

    def test_bf16_autocast_disables_with_warning(self):
        with self.assertWarns(UserWarning):
            with torch.autocast(device_type="rbln", dtype=torch.bfloat16):
                y = self._matmul_rbln()
        self.assertEqual(y.dtype, torch.float32)

    def test_enabled_false_force_fp32_block(self):
        # transformers' pervasive ``with autocast(enabled=False): # Force float32``.
        # Works independently of the advertised dtypes (value is not consulted here).
        with torch.autocast(device_type="rbln", enabled=False):
            y = self._matmul_rbln()
        self.assertEqual(y.dtype, torch.float32)


@pytest.mark.test_set_ci
class TestAutocastRealModelEager(TestCase):
    """A real (tiny) transformers LlamaForCausalLM run eagerly under an rbln autocast
    region must not crash and must match the plain fp32 path (autocast is a no-op)."""

    @staticmethod
    def _tiny_llama():
        transformers = pytest.importorskip("transformers")
        cfg = transformers.LlamaConfig(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
            dtype=torch.float32,
        )
        torch.manual_seed(0)
        return transformers.LlamaForCausalLM(cfg).eval()

    def test_llama_eager_autocast_matches_fp32(self):
        model = self._tiny_llama().to("rbln:0")
        ids = torch.randint(0, 256, (1, 8)).to("rbln:0")
        with torch.no_grad():
            plain = model(ids).logits.cpu()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # "Disabling autocast" is expected here
                with torch.autocast(device_type="rbln", dtype=torch.float16):
                    autocast_logits = model(ids).logits.cpu()
        # autocast disabled -> identical fp32 execution.
        self.assertEqual(autocast_logits.dtype, torch.float32)
        torch.testing.assert_close(autocast_logits, plain)


if __name__ == "__main__":
    run_tests()
