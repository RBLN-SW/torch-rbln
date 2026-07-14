# Owner(s): ["module: PrivateUse1"]

from unittest.mock import patch

import pytest
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

from test.utils import xfail_atom, xfail_rebel

from torch_rbln._internal.device_arch_utils import _arch_from_npu_name, is_atom_device, is_rebel_device


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


instantiate_device_type_tests(TestArchFromNpuName, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestArchPredicates, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestArchXfailMarkers, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
