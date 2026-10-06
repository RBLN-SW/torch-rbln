# Owner(s): ["module: PrivateUse1"]
"""A size torch.compile leaves open is a dynamic axis of the graph it compiles: tensors of any extent
there run that one graph, and capture_programs reports what each further index of the axis takes.
"""

import pytest
import rbln.ops  # noqa: F401  -- defines the rbln_custom_ops schemas
import torch
from torch.testing._internal.common_utils import run_tests, TestCase


HEADS, GROUPS, DIM, MAX_SEQ = 8, 2, 128, 2048
SCALE = DIM**-0.5


class Decode(torch.nn.Module):
    """One decode step of paged attention, which writes the key and value at `seq` of the block
    `table` names."""

    def forward(self, query, key, value, kcache, vcache, seq, table):
        scale = torch.tensor(SCALE)
        return torch.ops.rbln_custom_ops.paged_causal_attn_decode(
            query, key, value, kcache, vcache, seq, scale, table, MAX_SEQ
        )


def reference(query, key, value, kcache, vcache, start):
    kcache[..., start : start + 1, :] = key
    vcache[..., start : start + 1, :] = value
    keys, values = kcache[..., : start + 1, :], vcache[..., : start + 1, :]
    return torch.softmax((query @ keys.transpose(-1, -2)) * SCALE, -1) @ values


@pytest.mark.test_set_ci
class TestGraphDynamicInputs(TestCase):
    def _decode(self, compiled, blocks):
        query = torch.randn(1, HEADS, GROUPS, 1, DIM)
        key, value = (torch.randn(1, HEADS, 1, 1, DIM) for _ in range(2))
        kref, vref = (torch.randn(blocks, HEADS, 1, MAX_SEQ, DIM) / 10 for _ in range(2))
        kcache, vcache = kref.to("rbln"), vref.to("rbln")
        for cache in (kcache, vcache):
            torch._dynamo.mark_dynamic(cache, 0)
        seq = torch.tensor([[5]], dtype=torch.int32, device="rbln")
        table = torch.tensor([[blocks - 1]], dtype=torch.int16, device="rbln")
        expected = reference(query, key, value, kref[-1:].clone(), vref[-1:].clone(), 5)
        with torch.rbln.capture_programs() as programs:
            out = compiled(query.to("rbln"), key.to("rbln"), value.to("rbln"), kcache, vcache, seq, table)
        self.assertEqual(out.cpu(), expected, atol=5e-2, rtol=5e-2)
        self.assertEqual(kcache.cpu()[-1, :, :, 5], key[0, :, :, 0], atol=2e-2, rtol=2e-2)
        return programs

    def test_caches_of_any_extent_run_one_graph(self):
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        compiled = torch.compile(Decode(), backend="rbln", dynamic=False)
        for blocks in (4, 6):
            self._decode(compiled, blocks)
        self.assertEqual(torch._dynamo.utils.counters["stats"]["unique_graphs"], 1)

    def test_a_program_reports_the_bytes_of_each_step_of_the_axis(self):
        torch._dynamo.reset()
        (program,) = self._decode(torch.compile(Decode(), backend="rbln", dynamic=False), 4)
        (kcache,) = [spec for spec in program.input_specs if spec.name == "kcache"]
        self.assertEqual(kcache.shape, (4, HEADS, 1, MAX_SEQ, DIM))
        (axis,) = kcache.arg.logical.dynamic_axes
        self.assertEqual(axis.axis, 0)
        for shard in kcache.arg.shards:
            self.assertGreater(shard["step_nbytes"], 0)
            self.assertEqual(shard["min_nbytes"], axis.min * shard["step_nbytes"])


if __name__ == "__main__":
    run_tests()
