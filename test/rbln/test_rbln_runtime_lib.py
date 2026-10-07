# Owner(s): ["module: PrivateUse1"]

"""Tests for how torch_rbln._internal.rbln_runtime_lib finds the runtime it runs on.

The rule under test is that the runtime is the ``librebel_v2_rt.so`` the ``rebel.v2`` package maps:
nothing is searched for, the mapping is read back from ``/proc/self/maps``, and anything other
than exactly one mapped copy is an ImportError that says what to fix.
"""

import os
import sys
import types
from unittest.mock import patch

import pytest
from torch.testing._internal.common_utils import run_tests, TestCase

from torch_rbln._internal import rbln_runtime_lib


# Deselected by rebel_compiler's CI (`-m "not torch_rbln_only"`); see the marker in pyproject.
# How torch_rbln *finds* the runtime, not what the runtime does once loaded.
pytestmark = pytest.mark.torch_rbln_only


_RUNTIME = "/tree/build/rebel/v2/librebel_v2_rt.so"
_OTHER_RUNTIME = "/other/build/rebel/v2/librebel_v2_rt.so"


def _maps_line(path: str = "", perms: str = "r-xp") -> str:
    """One /proc/<pid>/maps line; the kernel pads the inode column before the path."""
    line = f"7f1234560000-7f1234570000 {perms} 00001000 fd:01 1234567"
    return f"{line}{' ' * 19}{path}\n" if path else f"{line} \n"


@pytest.mark.test_set_ci
class TestParseMappedLibraries(TestCase):
    """Reading which runtime is mapped out of /proc/<pid>/maps."""

    def test_every_segment_of_one_library_is_one_path(self):
        lines = [_maps_line(_RUNTIME, "r--p"), _maps_line(_RUNTIME, "r-xp"), _maps_line(_RUNTIME, "rw-p")]
        self.assertEqual(rbln_runtime_lib.parse_mapped_libraries(lines), [_RUNTIME])

    def test_other_files_and_anonymous_mappings_are_skipped(self):
        lines = [
            _maps_line(),
            _maps_line("[heap]"),
            _maps_line("/tree/build/rebel/v2/librebel_v2_artifact.so"),
            _maps_line("/tree/build/rebel/v2/librebel_v2_rt.so.bak"),
            _maps_line("/tree/rebel/python/rebel/v2/runtime/_runtime.cpython-312-x86_64-linux-gnu.so"),
            _maps_line(_RUNTIME),
        ]
        self.assertEqual(rbln_runtime_lib.parse_mapped_libraries(lines), [_RUNTIME])

    def test_copies_are_listed_in_mapping_order(self):
        lines = [_maps_line(_OTHER_RUNTIME), _maps_line(_RUNTIME), _maps_line(_OTHER_RUNTIME)]
        self.assertEqual(rbln_runtime_lib.parse_mapped_libraries(lines), [_OTHER_RUNTIME, _RUNTIME])

    def test_a_deleted_mapping_is_reported_as_a_path(self):
        # The kernel marks a mapping whose file is gone with " (deleted)". The marker is not part
        # of the path, and what this returns is handed to dlopen by the ABI check.
        lines = [_maps_line(f"{_RUNTIME} (deleted)")]
        self.assertEqual(rbln_runtime_lib.parse_mapped_libraries(lines), [_RUNTIME])

    def test_a_path_with_spaces_is_kept_whole(self):
        path = "/home/some user/tree/build/rebel/v2/librebel_v2_rt.so"
        self.assertEqual(rbln_runtime_lib.parse_mapped_libraries([_maps_line(path)]), [path])

    def test_another_name_can_be_looked_up(self):
        lines = [_maps_line(_RUNTIME), _maps_line("/tree/build/rebel/v2/librebel_v2_artifact.so")]
        self.assertEqual(
            rbln_runtime_lib.parse_mapped_libraries(lines, name="librebel_v2_artifact.so"),
            ["/tree/build/rebel/v2/librebel_v2_artifact.so"],
        )

    def test_an_unreadable_mapping_table_reads_as_nothing_mapped(self):
        with patch.object(rbln_runtime_lib, "_MAPS_PATH", "/nonexistent/maps"):
            self.assertEqual(rbln_runtime_lib.loaded_runtime_libraries(), [])


@pytest.mark.test_set_ci
class TestLoadRuntimeLibrary(TestCase):
    """Mapping the runtime through ``rebel.v2.runtime`` and reporting what was mapped."""

    _MODULE = types.SimpleNamespace(__file__="/tree/rebel/python/rebel/v2/runtime/__init__.py")

    def _mapped(self, paths: list[str]):
        return (
            patch.object(rbln_runtime_lib, "import_runtime_package", return_value=self._MODULE),
            patch.object(rbln_runtime_lib, "loaded_runtime_libraries", return_value=paths),
        )

    def test_the_one_mapped_runtime_is_returned(self):
        imported, mapped = self._mapped([_RUNTIME])
        with imported, mapped:
            self.assertEqual(rbln_runtime_lib.load_runtime_library(), _RUNTIME)

    def test_an_unimportable_rbln_says_where_to_get_it(self):
        # A None entry in sys.modules makes the import raise ImportError.
        with patch.dict(sys.modules, {"rebel.v2": None, "rebel.v2.runtime": None}):
            with self.assertRaises(ImportError) as ctx:
                rbln_runtime_lib.load_runtime_library()
        message = str(ctx.exception)
        self.assertIn("`import rebel.v2.runtime` failed", message)
        self.assertIn("PYTHONPATH=$REBEL_HOME/rebel/python", message)
        self.assertIn("python -m torch_rbln.diagnose", message)

    def test_an_import_that_maps_no_runtime_is_an_error(self):
        imported, mapped = self._mapped([])
        with imported, mapped:
            with self.assertRaises(ImportError) as ctx:
                rbln_runtime_lib.load_runtime_library()
        message = str(ctx.exception)
        self.assertIn("no librebel_v2_rt.so is mapped", message)
        self.assertIn(self._MODULE.__file__, message)

    def test_two_mapped_copies_are_an_error(self):
        # torch-rbln's libraries would bind to one of them and rbln possibly to the other: two
        # allocators and two device registries that cannot see each other.
        imported, mapped = self._mapped([_RUNTIME, _OTHER_RUNTIME])
        with imported, mapped:
            with self.assertRaises(ImportError) as ctx:
                rbln_runtime_lib.load_runtime_library()
        message = str(ctx.exception)
        self.assertIn(_RUNTIME, message)
        self.assertIn(_OTHER_RUNTIME, message)
        self.assertIn("share one runtime", message)


@pytest.mark.test_set_ci
class TestInstalledRuntime(TestCase):
    """The rebel.v2 package importable here, if any."""

    def test_rbln_maps_exactly_one_runtime_and_it_is_the_one_returned(self):
        try:
            path = rbln_runtime_lib.load_runtime_library()
        except ImportError as e:
            self.skipTest(f"the rebel.v2 runtime is not available: {e}")
        self.assertEqual(os.path.basename(path), rbln_runtime_lib.RUNTIME_LIB_NAME)
        self.assertTrue(os.path.isfile(path), path)
        self.assertEqual(rbln_runtime_lib.loaded_runtime_libraries(), [path])


if __name__ == "__main__":
    run_tests()
