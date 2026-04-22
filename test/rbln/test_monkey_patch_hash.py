# Owner(s): ["module: PrivateUse1"]

"""Tests for ``patch_get_torch_hash`` in ``torch_rbln._internal.monkey_patches``.

Workaround for FINE-542: rebel-compiler's ``get_torch_hash`` does not include
graph structure in its signature, so two parameter-less ``fx.GraphModule``s
collide on the same ``mod_name``. Under ``use_weight_sharing=True`` the second
compile fails with ``DEVICE_GRAPH_CONVERSION``. The patch forces
``include_graph=True`` so distinct graphs get distinct hashes.
"""

import pytest
import torch
import torch.nn as nn
import torch_rbln  # noqa: F401  -- registers backend, applies patches
from torch.testing._internal.common_utils import run_tests, TestCase

from torch_rbln._internal import monkey_patches as mp


def _capture_graph_module(fn, *example_inputs):
    """Capture the fx.GraphModule that dynamo would hand the rbln backend."""
    captured = {}

    def _spy(gm, _inputs):
        captured["gm"] = gm
        return gm.forward

    compiled = torch.compile(fn, backend=_spy, dynamic=False)
    compiled(*example_inputs)
    return captured["gm"]


@pytest.mark.test_set_ci
class TestPatchGetTorchHash(TestCase):
    def test_patch_marker_set(self):
        self.assertTrue(mp._get_torch_hash_patched)

    def test_patch_idempotent(self):
        mp.apply_all_patches()
        mp.apply_all_patches()
        self.assertTrue(mp._get_torch_hash_patched)

    def test_module_bindings_updated(self):
        """``compile_from_any`` imports ``get_torch_hash`` by name; that
        binding must point to the patched function so the cached reference
        in the importing module also goes through the include_graph path.
        """
        import rebel.compile_from_any as _cfa
        import rebel.core.compilation._impl as _impl
        self.assertIs(_cfa.get_torch_hash, _impl.get_torch_hash)
        # Optional: ``rebel.core.compilation`` may or may not re-export the
        # name depending on the wheel build; tolerate either.
        import rebel.core.compilation as _rc
        if getattr(_rc, "get_torch_hash", None) is not None:
            self.assertIs(_rc.get_torch_hash, _impl.get_torch_hash)

    def test_distinct_paramless_graphs_get_distinct_hashes(self):
        """Two structurally different parameter-less graphs must hash differently.

        Pre-patch (or upstream): both signatures are
            ("no-param", "no-buffer", "", version, "class:GraphModule", "tp:1")
        and produce identical hashes -> mod_name collision -> FINE-542 failure.
        Post-patch: ``include_graph=True`` differentiates them.
        """
        from rebel.core.compilation._impl import get_torch_hash

        gm_mul = _capture_graph_module(lambda x: x * 1024, torch.zeros(2, dtype=torch.int32))
        gm_softmax = _capture_graph_module(
            lambda x: torch.nn.functional.softmax(x, dim=-1),
            torch.zeros(1, 16, dtype=torch.float16),
        )

        # Sanity: same class name, both no-param.
        self.assertEqual(gm_mul.__class__.__name__, gm_softmax.__class__.__name__)
        self.assertEqual(list(gm_mul.parameters()), [])
        self.assertEqual(list(gm_softmax.parameters()), [])

        h_mul = get_torch_hash(gm_mul, tensor_parallel_size=1)
        h_softmax = get_torch_hash(gm_softmax, tensor_parallel_size=1)

        # The whole point of the patch: these MUST differ.
        self.assertNotEqual(
            h_mul,
            h_softmax,
            f"hash collision: both gm hashed to {h_mul!r} (FINE-542 unfixed)",
        )

    def test_same_graph_same_hash(self):
        """Idempotence: hashing the same graph twice yields the same value."""
        from rebel.core.compilation._impl import get_torch_hash

        gm = _capture_graph_module(lambda x: x + 1, torch.zeros(4, dtype=torch.float16))
        h1 = get_torch_hash(gm, tensor_parallel_size=1)
        h2 = get_torch_hash(gm, tensor_parallel_size=1)
        self.assertEqual(h1, h2)

    def test_tp_size_affects_hash(self):
        """Different tp_size must still differentiate (preserved upstream behavior)."""
        from rebel.core.compilation._impl import get_torch_hash

        gm = _capture_graph_module(lambda x: x + 1, torch.zeros(4, dtype=torch.float16))
        h1 = get_torch_hash(gm, tensor_parallel_size=1)
        h4 = get_torch_hash(gm, tensor_parallel_size=4)
        self.assertNotEqual(h1, h4)


if __name__ == "__main__":
    run_tests()
