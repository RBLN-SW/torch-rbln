# Owner(s): ["module: PrivateUse1"]
"""A graph that writes a tensor it is given in place writes the caller's tensor, wherever the
tensor lies: where its arg takes it, off the alignment the arg takes, or on the host.
"""

import os
from unittest import mock

import pytest
import rbln.ops  # noqa: F401  -- defines the rbln_custom_ops schemas
import torch
from torch.testing._internal.common_utils import run_tests, TestCase


SHAPE = (3, 2, 4, 64)


class CacheWrite(torch.nn.Module):
    """Writes twice x into block 1 of the cache it is given, as a decoder writes its KV cache."""

    def forward(self, x, block, cache):
        axis = torch.tensor(1, dtype=torch.int16)
        torch.ops.rbln_custom_ops.rbln_cache_update(cache, (x * 2.0).unsqueeze(1), block[0], axis)
        return x + 1.0


@pytest.mark.test_set_ci
class TestGraphWrittenInputs(TestCase):
    def _write(self, cache):
        torch._dynamo.reset()
        x = torch.randn(3, 4, 64, dtype=torch.float16)
        block = torch.tensor([1], dtype=torch.int16, device="rbln")
        out = torch.compile(CacheWrite(), backend="rbln", dynamic=False)(x.to("rbln"), block, cache)
        self.assertEqual(out.cpu(), x + 1.0, atol=5e-2, rtol=5e-2)
        written = cache.cpu()
        self.assertEqual(written[:, 1], 2.0 * x, atol=5e-2, rtol=5e-2)
        self.assertFalse(written[:, 0].any())

    def test_a_tensor_on_the_device_is_written(self):
        self._write(torch.zeros(SHAPE, dtype=torch.float16, device="rbln"))

    def test_a_view_off_the_alignment_is_written(self):
        flat = torch.zeros(1 + torch.Size(SHAPE).numel(), dtype=torch.float16, device="rbln")
        self._write(flat[1:].view(SHAPE))
        self.assertEqual(flat[0].item(), 0.0)

    def test_a_tensor_on_the_host_is_written(self):
        self._write(torch.zeros(SHAPE, dtype=torch.float16))

    def test_a_round_trip_through_the_host_raises_when_disabled(self):
        cache = torch.zeros(SHAPE, dtype=torch.float16, device="rbln")
        with mock.patch.dict(os.environ, {"TORCH_RBLN_DISABLE_FALLBACK": "host_round_trip"}):
            with self.assertRaisesRegex(RuntimeError, "input cache is written in place"):
                self._write(cache)


if __name__ == "__main__":
    run_tests()
