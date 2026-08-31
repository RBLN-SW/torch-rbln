# Owner(s): ["module: PrivateUse1"]

import sys
from unittest.mock import patch

import pytest
from torch.testing._internal.common_utils import run_tests, TestCase

from test.utils import xfail_atom, xfail_rebel

from torch_rbln._internal.device_arch_utils import _arch_from_npu_name, get_device_arch, is_atom_device, is_rebel_device


@pytest.mark.test_set_ci
class TestArchFromNpuName(TestCase):
    """Maps an NPU name to a device family."""

    def test_atom_name(self):
        self.assertEqual(_arch_from_npu_name("RBLN-CA12"), "atom")

    def test_rebel_name(self):
        self.assertEqual(_arch_from_npu_name("RBLN-CR03"), "rebel")

    def test_name_is_case_insensitive(self):
        self.assertEqual(_arch_from_npu_name("rbln-cr03"), "rebel")

    def test_empty_name(self):
        self.assertEqual(_arch_from_npu_name(""), "unknown")

    def test_unrecognized_name(self):
        self.assertEqual(_arch_from_npu_name("RBLN-XX99"), "unknown")


@pytest.mark.test_set_ci
class TestArchPredicates(TestCase):
    """``is_rebel_device`` / ``is_atom_device`` reflect ``get_device_arch``."""

    def test_is_rebel_device_true_on_rebel(self):
        with patch("torch_rbln._internal.device_arch_utils.get_device_arch", return_value="rebel"):
            self.assertTrue(is_rebel_device())

    def test_is_rebel_device_false_on_atom(self):
        with patch("torch_rbln._internal.device_arch_utils.get_device_arch", return_value="atom"):
            self.assertFalse(is_rebel_device())

    def test_is_atom_device_true_on_atom(self):
        with patch("torch_rbln._internal.device_arch_utils.get_device_arch", return_value="atom"):
            self.assertTrue(is_atom_device())

    def test_is_atom_device_false_on_rebel(self):
        with patch("torch_rbln._internal.device_arch_utils.get_device_arch", return_value="rebel"):
            self.assertFalse(is_atom_device())


@pytest.mark.test_set_ci
class TestArchXfailMarkers(TestCase):
    """xfail markers set ``condition`` per arch and ``strict=True``."""

    def test_xfail_rebel_active_on_rebel(self):
        with patch("torch_rbln._internal.device_arch_utils.get_device_arch", return_value="rebel"):
            mark = xfail_rebel("REBEL: feature not yet supported")
        self.assertTrue(mark.mark.kwargs["condition"])
        self.assertTrue(mark.mark.kwargs["strict"])
        self.assertEqual(mark.mark.kwargs["reason"], "REBEL: feature not yet supported")

    def test_xfail_rebel_inert_on_atom(self):
        with patch("torch_rbln._internal.device_arch_utils.get_device_arch", return_value="atom"):
            mark = xfail_rebel("REBEL: feature not yet supported")
        self.assertFalse(mark.mark.kwargs["condition"])

    def test_xfail_atom_active_on_atom(self):
        with patch("torch_rbln._internal.device_arch_utils.get_device_arch", return_value="atom"):
            mark = xfail_atom("ATOM: feature not yet supported")
        self.assertTrue(mark.mark.kwargs["condition"])
        self.assertTrue(mark.mark.kwargs["strict"])

    def test_xfail_atom_inert_on_rebel(self):
        with patch("torch_rbln._internal.device_arch_utils.get_device_arch", return_value="rebel"):
            mark = xfail_atom("ATOM: feature not yet supported")
        self.assertFalse(mark.mark.kwargs["condition"])


@pytest.mark.test_set_ci
class TestArchResolutionFailure(TestCase):
    """A host with no NPU and a moved ``get_npu_name`` must not look alike.

    Both used to answer ``"unknown"``, and every caller of ``get_device_arch``
    is an architecture gate, so the second one turns all of them off at once
    with nothing failing. Only the first is a legitimate ``"unknown"``.
    """

    def setUp(self) -> None:
        get_device_arch.cache_clear()
        self.addCleanup(get_device_arch.cache_clear)

    def test_no_device_answers_unknown(self) -> None:
        # What rebel returns for an index no device claims.
        with patch("rebel.device_info.get_npu_name", return_value=None):
            self.assertEqual(get_device_arch(), "unknown")

    def test_moved_entry_point_raises(self) -> None:
        # None in sys.modules is what CPython turns a moved module into.
        with patch.dict(sys.modules, {"rebel.device_info": None}):
            with self.assertRaises(ImportError):
                get_device_arch()


if __name__ == "__main__":
    run_tests()
