# Owner(s): ["module: PrivateUse1"]
"""A graph that writes a tensor in place which the device holds otherwise than torch, such as a float16
cache it keeps in dlfloat16, keeps the tensor as the device holds it: later calls, of that graph or of
another holding it alike, bind it in place, and whatever else reaches it gets it back as torch holds it.
"""

import pytest
import rbln.ops  # noqa: F401  -- defines the rbln_custom_ops schemas
import torch
from torch.testing._internal.common_utils import run_tests, TestCase


SHAPE = (3, 4, 4, 64)


class CacheWrite(torch.nn.Module):
    """Writes `scale` times x into block `block` of the cache it is given, as a decoder writes its KV cache."""

    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x, block, cache):
        axis = torch.tensor(1, dtype=torch.int16)
        torch.ops.rbln_custom_ops.rbln_cache_update(cache, (x * self.scale).unsqueeze(1), block[0], axis)
        return x + 1.0


def _count(region, site):
    return region.dump()["hidden_host_bounce"]["by_site"][site]["count"]


@pytest.mark.test_set_ci
class TestGraphHeldInputs(TestCase):
    def setUp(self):
        torch._dynamo.reset()
        self.compiled = torch.compile(CacheWrite(2.0), backend="rbln", dynamic=False)
        self.cache = torch.zeros(SHAPE, dtype=torch.float16, device="rbln")
        self.x = torch.randn(SHAPE[0], *SHAPE[2:], dtype=torch.float16)

    def _write(self, compiled, x, block):
        index = torch.tensor([block], dtype=torch.int16, device="rbln")
        out = compiled(x.to("rbln"), index, self.cache)
        self.assertEqual(out.cpu(), x + 1.0, atol=5e-2, rtol=5e-2)

    def _expect(self, blocks):
        cache = self.cache.cpu()
        for block in range(SHAPE[1]):
            expected = blocks.get(block, torch.zeros_like(self.x))
            self.assertEqual(cache[:, block], expected, atol=5e-2, rtol=5e-2)

    def test_later_calls_bind_the_cache_in_place(self):
        self._write(self.compiled, self.x, 0)
        with torch.rbln.explain() as region:
            self._write(self.compiled, self.x * 3, 1)
        self.assertEqual(_count(region, "op_arg_through_host"), 0)
        self.assertEqual(_count(region, "held_tensor_released"), 0)
        self._expect({0: 2 * self.x, 1: 6 * self.x})

    def test_reading_the_cache_puts_it_back_until_the_next_call(self):
        self._write(self.compiled, self.x, 0)
        with torch.rbln.explain() as region:
            self._expect({0: 2 * self.x})
        self.assertEqual(_count(region, "held_tensor_released"), 1)
        self._write(self.compiled, self.x, 1)
        self._expect({0: 2 * self.x, 1: 2 * self.x})

    def test_an_eager_write_lands_in_the_held_cache(self):
        self._write(self.compiled, self.x, 0)
        y = torch.randn_like(self.x)
        self.cache[:, 2] = y.to("rbln")
        self._write(self.compiled, self.x, 1)
        self._expect({0: 2 * self.x, 1: 2 * self.x, 2: y})

    def test_another_graph_holding_it_alike_binds_it_in_place(self):
        self._write(self.compiled, self.x, 0)
        other = torch.compile(CacheWrite(-1.0), backend="rbln", dynamic=False)
        self._write(other, self.x, 1)
        with torch.rbln.explain() as region:
            self._write(other, self.x, 2)
            self._write(self.compiled, self.x, 3)
        self.assertEqual(_count(region, "op_arg_through_host"), 0)
        self.assertEqual(_count(region, "held_tensor_released"), 0)
        self._expect({0: 2 * self.x, 1: -self.x, 2: -self.x, 3: 2 * self.x})

    def test_memory_of_a_freed_cache_is_not_held(self):
        self._write(self.compiled, self.x, 0)
        self.cache = None
        with torch.rbln.explain() as region:
            reused = torch.ones(SHAPE, dtype=torch.float16, device="rbln")
            self.assertTrue(bool((reused.cpu() == 1).all()))
        self.assertEqual(_count(region, "held_tensor_released"), 0)


if __name__ == "__main__":
    run_tests()
