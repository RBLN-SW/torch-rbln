# Owner(s): ["module: PrivateUse1"]

"""Tests for ``torch.rbln.offload``.

``offload`` is a thin context manager over
``rebel._C.vmem.set_file_offloading_enabled`` that toggles a process-wide flag
controlling whether new RBLN tensor user views are routed to on-disk storage
or in-memory. Correctness of the underlying file offloading behavior is covered
by rebel-compiler's C++ tests (``vmem_file_offloading_test.cc``); these tests
verify the torch-rbln level exposure, the depth-counted nesting of the context
manager, and that allocations inside the block actually carry an on-file user
view (observed via ``rebel._C.vmem.debug.get_internal_states`` JSON).

The toggled flag and the depth counter are both process-local (each pytest-xdist
worker is its own process, with its own ``VMemoryManager`` and its own random
``tmpdir`` for offload files), and the JSON we observe is read out of that same
process — no cross-worker state leaks. So this class does not need
``single_worker``.
"""

import json

import pytest
import torch
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_rbln  # noqa: F401  -- registers the rbln device module
import torch_rbln.memory as torch_rbln_memory


def _user_view_kind(tensor: torch.Tensor) -> str:
    """Return the user-view kind ("on_memory" or "on_file") for an RBLN tensor.

    Reads ``rebel._C.vmem.debug.get_internal_states`` JSON for the tensor's vaddr
    (which equals ``tensor.data_ptr()`` for RBLN tensors) and returns whichever
    of the two known sub-keys the ``user_view`` carries.
    """
    from rebel._C import vmem as _rebel_vmem

    vaddr = tensor.data_ptr()
    raw = _rebel_vmem.debug.get_internal_states(vaddr)
    parsed = json.loads(raw)
    user_view = parsed["user_view"]
    if "on_file" in user_view:
        return "on_file"
    if "on_memory" in user_view:
        return "on_memory"
    raise AssertionError(f"Unexpected user_view shape: {user_view!r}")


def _make_tensor(device: str) -> torch.Tensor:
    """Allocate a small RBLN tensor and force the user view to materialize.

    A bare ``torch.zeros(..., device="rbln")`` keeps the entry in
    ``EMPTY_INIT_WITH_ZERO`` state with no ``user_view`` key. Copying real data
    in triggers H2V, which lands the entry in either ``on_memory`` or ``on_file``
    based on the current file offloading flag — that is the state we observe.
    """
    tensor = torch.zeros(8, dtype=torch.float32, device=device)
    tensor.copy_(torch.ones(8, dtype=torch.float32))
    return tensor


@pytest.mark.test_set_ci
class TestFileOffloading(TestCase):
    def setUp(self):
        self.device = "rbln:0"

    def tearDown(self):
        # Defensive: ensure file offloading is OFF and the depth counter is reset
        # so a failing test doesn't leak state into the next test (or test file).
        torch_rbln_memory._offload_depth = 0
        try:
            from rebel._C import vmem as _rebel_vmem

            _rebel_vmem.set_file_offloading_enabled(False)
        except Exception:
            pass
        try:
            torch.rbln.empty_cache(self.device)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Smoke / exposure
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
    # Behavior verification
    # ------------------------------------------------------------------
    def test_offload_toggles_user_view_routing(self):
        """Outside the CM allocations are on_memory; inside they are on_file."""
        outside_before = _make_tensor(self.device)
        self.assertEqual(_user_view_kind(outside_before), "on_memory")

        with torch.rbln.offload():
            inside = _make_tensor(self.device)
            self.assertEqual(_user_view_kind(inside), "on_file")

        outside_after = _make_tensor(self.device)
        self.assertEqual(_user_view_kind(outside_after), "on_memory")

    def test_offload_does_not_mutate_existing_tensor(self):
        """Toggling does not migrate existing user views; it affects new ones."""
        tensor_before = _make_tensor(self.device)
        self.assertEqual(_user_view_kind(tensor_before), "on_memory")

        with torch.rbln.offload():
            # Existing tensor's user view stays put.
            self.assertEqual(_user_view_kind(tensor_before), "on_memory")
            # New allocation is routed to file.
            new_tensor = _make_tensor(self.device)
            self.assertEqual(_user_view_kind(new_tensor), "on_file")

    def test_offload_nested(self):
        """Nested ``offload`` blocks track depth and disable on outermost exit."""
        self.assertEqual(torch_rbln_memory._offload_depth, 0)

        with torch.rbln.offload():
            self.assertEqual(torch_rbln_memory._offload_depth, 1)
            inner_outer = _make_tensor(self.device)
            self.assertEqual(_user_view_kind(inner_outer), "on_file")

            with torch.rbln.offload():
                self.assertEqual(torch_rbln_memory._offload_depth, 2)
                deepest = _make_tensor(self.device)
                self.assertEqual(_user_view_kind(deepest), "on_file")

            # After inner exit, still inside the outer CM: stays on_file.
            self.assertEqual(torch_rbln_memory._offload_depth, 1)
            after_inner = _make_tensor(self.device)
            self.assertEqual(_user_view_kind(after_inner), "on_file")

        self.assertEqual(torch_rbln_memory._offload_depth, 0)
        after_all = _make_tensor(self.device)
        self.assertEqual(_user_view_kind(after_all), "on_memory")

    def test_offload_restores_state_on_exception(self):
        """Exception inside the block must still flip offloading back off."""
        with self.assertRaises(RuntimeError):
            with torch.rbln.offload():
                self.assertEqual(torch_rbln_memory._offload_depth, 1)
                raise RuntimeError("boom")

        self.assertEqual(torch_rbln_memory._offload_depth, 0)
        post = _make_tensor(self.device)
        self.assertEqual(_user_view_kind(post), "on_memory")


if __name__ == "__main__":
    run_tests()
