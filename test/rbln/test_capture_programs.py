# Owner(s): ["module: PrivateUse1"]

"""Tests for ``torch.rbln.capture_programs``.

The scope itself (ordering, nesting, thread-locality) is rebel-compiler's and is
tested there (``tests/python/test_rebel/test_program_capture.py``). These tests cover
what is observable from torch-rbln:

* the surface (``torch.rbln.capture_programs`` / ``CompiledProgram`` are exposed and
  are the rebel objects, so there is one implementation);
* a torch.compile on the rbln backend inside the scope yields one program per graph
  the backend built, carrying the runtime behind the compiled callable.
"""

import pytest
import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln.programs as torch_rbln_programs


@pytest.mark.test_set_ci
class TestCaptureProgramsSurface(TestCase):
    def test_exposed_on_torch_rbln(self):
        from rebel.core.program_capture import capture_programs, CompiledProgram

        self.assertIn("capture_programs", torch_rbln_programs.__all__)
        self.assertIn("CompiledProgram", torch_rbln_programs.__all__)
        self.assertIs(torch.rbln.capture_programs, capture_programs)
        self.assertIs(torch.rbln.CompiledProgram, CompiledProgram)

    def test_empty_scope_yields_empty_list(self):
        with torch.rbln.capture_programs() as programs:
            pass
        self.assertEqual(programs, [])


@pytest.mark.test_set_ci
class TestCaptureProgramsCompile(TestCase):
    def test_one_program_per_backend_build(self, device):
        from rebel.sync_runtime import DynamoRuntime

        class AddModule(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        x = torch.ones((1, 3, 8, 8), dtype=torch.float16, device=device)
        y = torch.ones((1, 3, 8, 8), dtype=torch.float16, device=device)

        with torch.rbln.capture_programs() as programs:
            compiled = torch.compile(AddModule().eval(), backend="rbln", dynamic=False)
            out = compiled(x, y)
            compiled(x, y)  # dynamo cache hit: the backend does not run again

        self.assertEqual(out.cpu(), torch.full((1, 3, 8, 8), 2.0, dtype=torch.float16))
        self.assertEqual(len(programs), 1)
        program = programs[0]
        self.assertIsInstance(program, torch.rbln.CompiledProgram)
        self.assertIsInstance(program.runtime, DynamoRuntime)
        self.assertEqual(program.device, x.device)
        self.assertTrue(program.name)  # dynamo compile id, e.g. "0/0"

    def test_programs_outside_scope_are_not_recorded(self, device):
        compiled = torch.compile(lambda t: t * 2, backend="rbln", dynamic=False)
        compiled(torch.ones(4, device=device))  # built before the scope opens

        with torch.rbln.capture_programs() as programs:
            compiled(torch.ones(4, device=device))  # cache hit, no new build

        self.assertEqual(programs, [])


instantiate_device_type_tests(TestCaptureProgramsCompile, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
