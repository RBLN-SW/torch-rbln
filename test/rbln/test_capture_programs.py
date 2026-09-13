# Owner(s): ["module: PrivateUse1"]

"""Tests for ``torch.rbln.capture_programs``.

The scope itself (ordering, nesting, thread-locality) is rebel-compiler's and is
tested there (``tests/python/test_rebel/test_program_capture.py``). These tests cover
what is observable from torch-rbln:

* the surface (``torch.rbln.capture_programs`` / ``CompiledProgram`` are exposed and
  are the rebel objects, so there is one implementation);
* the surface staying lazy, so ``import torch`` does not pull ``rebel`` in through the
  autoload hook;
* a torch.compile on the rbln backend inside the scope yields one program per graph
  the backend built, carrying the runtime behind the compiled callable.
"""

import os
import subprocess
import sys
import textwrap

import pytest
import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln.programs as torch_rbln_programs


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
class TestCaptureProgramsLazy(TestCase):
    def test_import_does_not_load_rebel(self):
        """Importing torch_rbln must leave ``rebel`` out of ``sys.modules``.

        ``import torch`` runs ``import torch_rbln`` through the autoload hook, so a
        top-level rebel import here puts the dynamo backend and its custom op
        registration on every torch user's import path. Both names in
        ``torch_rbln.programs`` therefore resolve on first access instead.
        """
        script = f"""
            import sys
            sys.path.insert(0, {_PROJECT_ROOT!r})
            import torch, torch_rbln  # noqa: F401
            loaded = sorted(m for m in sys.modules if m == "rebel" or m.startswith("rebel."))
            assert not loaded, "import pulled in " + ", ".join(loaded)

            from rebel.core.program_capture import capture_programs
            assert torch.rbln.capture_programs is capture_programs
            print("LAZY_OK")
        """
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertTrue(
            result.returncode == 0 and "LAZY_OK" in result.stdout,
            f"torch_rbln imported rebel eagerly\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}",
        )


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
