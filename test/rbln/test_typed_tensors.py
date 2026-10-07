# Owner(s): ["module: PrivateUse1"]
"""A tensor made in the type of a compiled program's arg, such as a float16 cache a graph keeps in
dlfloat16, keeps that type for its life: graphs of the type bind it in place, copies move its
elements to and from other types, and nothing reads its bytes as torch holds a tensor.
"""

import pytest
import rebel.v2.ops  # noqa: F401  -- defines the rbln_custom_ops schemas
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


def _compile(scale):
    """The compiled CacheWrite of `scale`, and the arg its cache binds to, from a first call over a
    cache torch holds, as a model runner profiles before it makes its caches."""
    compiled = torch.compile(CacheWrite(scale), backend="rbln", dynamic=False)
    with torch.rbln.capture_programs() as programs:
        compiled(
            torch.zeros(SHAPE[0], *SHAPE[2:], dtype=torch.float16, device="rbln"),
            torch.tensor([0], dtype=torch.int16, device="rbln"),
            torch.zeros(SHAPE, dtype=torch.float16, device="rbln"),
        )
    (program,) = programs
    return compiled, program.input_specs[-1].arg


@pytest.mark.test_set_ci
class TestTypedTensors(TestCase):
    def setUp(self):
        torch._dynamo.reset()
        self.compiled, self.arg = _compile(2.0)
        self.cache = torch.rbln.zeros_typed(self.arg)
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

    def test_the_arg_holds_the_cache_otherwise_than_torch(self):
        self.assertNotEqual(self.arg.transform, "")
        self.assertEqual(self.cache.dtype, torch.float16)
        self.assertEqual(tuple(self.cache.shape), SHAPE)

    def test_every_call_binds_the_cache_in_place(self):
        with torch.rbln.explain() as region:
            self._write(self.compiled, self.x, 0)
            self._write(self.compiled, self.x * 3, 1)
        self.assertEqual(_count(region, "op_arg_through_host"), 0)
        self.assertEqual(_count(region, "typed_through_host"), 0)
        self._expect({0: 2 * self.x, 1: 6 * self.x})

    def test_another_graph_of_the_type_binds_it_in_place(self):
        other, arg = _compile(-1.0)
        self.assertEqual(arg.type_id, self.arg.type_id)
        with torch.rbln.explain() as region:
            self._write(self.compiled, self.x, 0)
            self._write(other, self.x, 1)
        self.assertEqual(_count(region, "op_arg_through_host"), 0)
        self.assertEqual(_count(region, "typed_through_host"), 0)
        self._expect({0: 2 * self.x, 1: -self.x})

    def test_blocks_cross_to_and_from_the_cpu_as_their_elements(self):
        # A KV connector stages blocks through host buffers this way.
        self._write(self.compiled, self.x, 0)
        read = torch.empty_like(self.x)
        y = torch.randn_like(self.x)
        with torch.rbln.explain() as region:
            torch._foreach_copy_([read], [self.cache[:, 0]])
            torch._foreach_copy_([self.cache[:, 3]], [y])
            self.cache[:, 2].copy_(y)
        self.assertEqual(_count(region, "typed_through_host"), 0)
        self.assertEqual(read, 2 * self.x, atol=5e-2, rtol=5e-2)
        self._write(self.compiled, self.x, 1)
        self._expect({0: 2 * self.x, 1: 2 * self.x, 2: y, 3: y})

    def test_blocks_copied_within_the_type_move_as_they_are(self):
        # vLLM's prefix caching copies the tokens of one KV block into another.
        self._write(self.compiled, self.x, 0)
        with torch.rbln.explain() as region:
            torch._foreach_copy_([self.cache[:, 2]], [self.cache[:, 0]])
            self.cache[:, 3].copy_(self.cache[:, 0])
        self.assertEqual(_count(region, "typed_through_host"), 0)
        self._write(self.compiled, self.x, 1)
        self._expect({0: 2 * self.x, 1: 2 * self.x, 2: 2 * self.x, 3: 2 * self.x})

    def test_a_device_tensor_of_another_type_crosses_the_host(self):
        self._write(self.compiled, self.x, 0)
        y = torch.randn_like(self.x)
        with torch.rbln.explain() as region:
            self.cache[:, 2] = y.to("rbln")
            copied = self.cache.clone()
        self.assertEqual(_count(region, "typed_through_host"), 2)
        self._write(self.compiled, self.x, 1)
        self._expect({0: 2 * self.x, 1: 2 * self.x, 2: y})
        self.assertEqual(copied[:, 2].cpu(), y, atol=5e-2, rtol=5e-2)

    def test_an_eager_op_runs_on_a_copy_and_writes_back(self):
        self._write(self.compiled, self.x, 0)
        with torch.rbln.explain() as region:
            self.cache.mul_(0.5)
        self.assertGreater(_count(region, "typed_through_host"), 0)
        with torch.rbln.explain() as region:
            self._write(self.compiled, self.x, 1)
        self.assertEqual(_count(region, "op_arg_through_host"), 0)
        self._expect({0: self.x, 1: 2 * self.x})

    def test_zeroing_zeroes_the_elements(self):
        self._write(self.compiled, self.x, 0)
        self.cache.zero_()
        self._expect({})

    def test_reading_its_bytes_as_another_dtype_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "type of arg"):
            self.cache.view(torch.int16).cpu()

    def test_an_arg_torch_holds_as_it_is_takes_a_plain_tensor(self):
        with torch.rbln.capture_programs() as programs:
            torch.compile(lambda a: a * 2, backend="rbln", dynamic=False)(torch.ones(4, 64, device="rbln"))
        (program,) = programs
        plain = torch.rbln.empty_typed(program.input_specs[0].arg)
        self.assertEqual(program.input_specs[0].arg.transform, "")
        self.assertEqual(plain.view(torch.int32).cpu().shape, (4, 64))

    def test_a_cache_made_after_a_compile_only_pass_binds_from_the_first_run(self):
        # A model runner compiles its graphs over an interim cache, then makes the cache they take.
        torch._dynamo.reset()
        compiled = torch.compile(CacheWrite(2.0), backend="rbln", dynamic=False)
        interim = torch.zeros(SHAPE, dtype=torch.float16, device="rbln")
        index = torch.tensor([0], dtype=torch.int16, device="rbln")
        with torch.rbln.capture_programs() as programs, torch.rbln.explain() as region:
            with torch.rbln.compile_only():
                out = compiled(self.x.to("rbln"), index, interim)
        self.assertEqual(out.cpu(), torch.zeros_like(self.x))
        self.assertEqual(_count(region, "op_arg_through_host"), 0)
        (program,) = programs
        spec = program.input_specs[-1]
        self.assertEqual(spec.data_ptr, interim.untyped_storage().data_ptr())
        del interim
        self.cache = torch.rbln.zeros_typed(spec.arg)
        graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        with torch.rbln.explain() as region:
            self._write(compiled, self.x, 0)
        self.assertEqual(_count(region, "op_arg_through_host"), 0)
        self.assertEqual(torch._dynamo.utils.counters["stats"]["unique_graphs"], graphs)
        self._expect({0: 2 * self.x})

    def test_memory_of_a_freed_cache_has_no_type(self):
        self._write(self.compiled, self.x, 0)
        self.cache = None
        reused = torch.ones(SHAPE, dtype=torch.float16, device="rbln")
        self.assertTrue(bool((reused.cpu() == 1).all()))


if __name__ == "__main__":
    run_tests()
