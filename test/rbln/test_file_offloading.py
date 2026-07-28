# Owner(s): ["module: PrivateUse1"]

"""Tests for ``torch.rbln.offload``.

``offload`` is a thin context manager that toggles a process-wide flag
controlling whether new RBLN tensor user views are routed to on-disk storage
or in-memory. The underlying flag itself, and the migration of user views
between the two backings, are exercised by rebel-compiler's C++ tests
(``vmem_file_offloading_test.cc``).

These tests deliberately do not reach into ``rebel._C.vmem`` (which is a
rebel-compiler internal that should not be considered a public surface). They
verify only what is observable from the torch-rbln level:

* the surface (``torch.rbln.offload`` is a context manager and is in
  ``torch_rbln.memory.__all__``);
* the depth-counted nesting of the context manager (via the module-level
  ``_offload_depth``);
* exception-safe restoration of the off state.

The flag and the depth counter are both process-local, so each pytest-xdist
worker is isolated; the class does not need ``single_worker``.
"""

import pytest
import torch
from torch.testing._internal.common_utils import run_tests, TestCase

# torch.rbln is registered automatically via the "torch.backends" autoload
# entry point declared in torch-rbln's pyproject.toml, so importing torch is
# enough to expose torch.rbln.offload. We still import torch_rbln.memory
# directly to peek at its module-level _offload_depth from the nesting test,
# and torch_rbln._C for the tearDown safety net that bypasses the depth
# counter.
import torch_rbln._C as torch_rbln_C
import torch_rbln.memory as torch_rbln_memory


@pytest.mark.test_set_ci
class TestFileOffloading(TestCase):
    def tearDown(self):
        # Defensive: ensure the depth counter is reset and the underlying flag
        # is flipped off so a failing test doesn't leak state into the next
        # test (or the next test file).
        torch_rbln_memory._offload_depth = 0
        try:
            torch_rbln_C._set_file_offloading_enabled(False)  # noqa: SLF001
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Surface
    # ------------------------------------------------------------------
    def test_offload_context_exposed(self):
        """``offload`` is reachable on every documented path and is a CM."""
        from torch_rbln.memory import offload  # noqa: F401

        self.assertIn("offload", torch_rbln_memory.__all__)
        self.assertTrue(hasattr(torch.rbln, "offload"))
        cm = torch.rbln.offload()
        self.assertTrue(hasattr(cm, "__enter__"))
        self.assertTrue(hasattr(cm, "__exit__"))
        # Drive the protocol once to make sure entering/leaving works.
        with cm:
            pass

    # ------------------------------------------------------------------
    # Nesting (observable via the module-level depth counter)
    # ------------------------------------------------------------------
    def test_offload_nested_depth_tracking(self):
        """Nested ``offload`` blocks update the depth counter symmetrically."""
        self.assertEqual(torch_rbln_memory._offload_depth, 0)

        with torch.rbln.offload():
            self.assertEqual(torch_rbln_memory._offload_depth, 1)

            with torch.rbln.offload():
                self.assertEqual(torch_rbln_memory._offload_depth, 2)

            # After inner exit, still inside the outer CM.
            self.assertEqual(torch_rbln_memory._offload_depth, 1)

        self.assertEqual(torch_rbln_memory._offload_depth, 0)

    # ------------------------------------------------------------------
    # Exception safety
    # ------------------------------------------------------------------
    def test_offload_restores_depth_on_exception(self):
        """Exception inside the block must still decrement the depth counter."""
        with self.assertRaises(RuntimeError):
            with torch.rbln.offload():
                self.assertEqual(torch_rbln_memory._offload_depth, 1)
                raise RuntimeError("boom")

        self.assertEqual(torch_rbln_memory._offload_depth, 0)

    # ------------------------------------------------------------------
    # Temp storage release
    # ------------------------------------------------------------------
    def test_release_offload_temp_storage_surface(self):
        """``release_offload_temp_storage`` is exposed and reports a file count."""
        from torch_rbln.memory import release_offload_temp_storage  # noqa: F401

        self.assertIn("release_offload_temp_storage", torch_rbln_memory.__all__)
        self.assertTrue(hasattr(torch.rbln, "release_offload_temp_storage"))

        num_removed = torch.rbln.release_offload_temp_storage()
        self.assertIsInstance(num_removed, int)
        self.assertGreaterEqual(num_removed, 0)

    def test_release_offload_temp_storage_is_idempotent(self):
        """Releasing twice is safe; the second call has nothing left to remove."""
        torch.rbln.release_offload_temp_storage()
        self.assertEqual(torch.rbln.release_offload_temp_storage(), 0)

    def test_offload_usable_after_release(self):
        """A release does not wedge the flag: offload() still works afterwards."""
        torch.rbln.release_offload_temp_storage()

        with torch.rbln.offload():
            self.assertEqual(torch_rbln_memory._offload_depth, 1)

        self.assertEqual(torch_rbln_memory._offload_depth, 0)


if __name__ == "__main__":
    run_tests()
