# Owner(s): ["module: PrivateUse1"]

"""Tests for ``torch.rbln.capture_programs``.

These tests cover:

* the surface (``torch.rbln.capture_programs`` / ``CompiledProgram`` are exposed and
  are the ``torch_rbln.programs`` objects);
* the scope: nesting and thread-locality;
* ``import torch_rbln`` leaving the ``rebel`` package out;
* a torch.compile on the rbln backend inside the scope yields one program per graph
  the backend built, carrying the function behind the compiled callable and IO specs
  that mirror the graph's inputs and outputs.
"""

import os
import subprocess
import sys
import textwrap
import threading

import pytest
import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln.programs as torch_rbln_programs


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.mark.test_set_ci
class TestCaptureProgramsSurface(TestCase):
    def test_exposed_on_torch_rbln(self):
        for name in torch_rbln_programs.__all__:
            self.assertIs(getattr(torch.rbln, name), getattr(torch_rbln_programs, name))
        self.assertEqual(
            set(torch_rbln_programs.__all__),
            {"capture_programs", "CompiledProgram", "InputSpec", "OutputSpec"},
        )

    def test_empty_scope_yields_empty_list(self):
        with torch.rbln.capture_programs() as programs:
            pass
        self.assertEqual(programs, [])

    def test_nested_scopes_see_what_is_built_inside_them(self):
        with torch.rbln.capture_programs() as outer:
            torch_rbln_programs.submit_program("first")
            with torch.rbln.capture_programs() as inner:
                torch_rbln_programs.submit_program("second")
        self.assertEqual(outer, ["first", "second"])
        self.assertEqual(inner, ["second"])

    def test_scope_is_thread_local(self):
        with torch.rbln.capture_programs() as programs:
            thread = threading.Thread(target=torch_rbln_programs.submit_program, args=("elsewhere",))
            thread.start()
            thread.join()
        self.assertEqual(programs, [])


@pytest.mark.test_set_ci
class TestImportLeavesV1Out(TestCase):
    def test_import_does_not_load_v1(self):
        """Importing torch_rbln loads rebel.v2 alone: no other ``rebel`` module, and no TVM."""
        script = f"""
            import sys
            sys.path.insert(0, {_PROJECT_ROOT!r})
            import torch, torch_rbln  # noqa: F401
            loaded = sorted(
                m
                for m in sys.modules
                if m.split(".")[0] == "tvm"
                or (m.startswith("rebel.") and m != "rebel.v2" and not m.startswith("rebel.v2."))
            )
            assert not loaded, "import pulled in " + ", ".join(loaded)
            print("NO_V1")
        """
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertTrue(
            result.returncode == 0 and "NO_V1" in result.stdout,
            f"torch_rbln imported v1\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}",
        )


@pytest.mark.test_set_ci
class TestCaptureProgramsCompile(TestCase):
    def test_one_program_per_backend_build(self, device):
        from rebel import v2

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
        self.assertIsInstance(program.function, v2.Function)
        self.assertEqual(program.device, x.device)
        self.assertTrue(program.name)  # dynamo compile id, e.g. "0/0"

        # The IO specs mirror the graph: two named float16 inputs and one output, each an arg
        # of the function.
        self.assertEqual([spec.shape for spec in program.input_specs], [(1, 3, 8, 8), (1, 3, 8, 8)])
        self.assertEqual([spec.dtype for spec in program.input_specs], [torch.float16, torch.float16])
        self.assertTrue(all(spec.name for spec in program.input_specs))
        self.assertEqual([spec.shape for spec in program.output_specs], [(1, 3, 8, 8)])
        self.assertEqual(program.output_specs[0].dtype, torch.float16)
        for spec in (*program.input_specs, *program.output_specs):
            self.assertIsInstance(spec, (torch.rbln.InputSpec, torch.rbln.OutputSpec))
            self.assertEqual(tuple(spec.arg.logical.shape), spec.shape)
            self.assertEqual(len(spec.arg.shards), 1)

    def test_programs_outside_scope_are_not_recorded(self, device):
        compiled = torch.compile(lambda t: t * 2, backend="rbln", dynamic=False)
        compiled(torch.ones(4, device=device))  # built before the scope opens

        with torch.rbln.capture_programs() as programs:
            compiled(torch.ones(4, device=device))  # cache hit, no new build

        self.assertEqual(programs, [])


instantiate_device_type_tests(TestCaptureProgramsCompile, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
