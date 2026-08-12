# Owner(s): ["module: PrivateUse1"]

"""Tests for how torch_rbln._internal.rbln_runtime_lib locates librbln.so.

The rule under test is that the rebel-compiler distribution's record of what it installed
decides, with LD_LIBRARY_PATH as the one override, and that nothing guesses at a layout ahead
of that record.
"""

import os
import shutil
import tempfile
from unittest.mock import patch

import pytest
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

from torch_rbln._internal import rbln_runtime_lib


def _record(name: str, located: str):
    """A distribution record entry: a relative name that can also locate an absolute path."""

    class _Entry(str):
        def locate(self) -> str:
            return located

    return _Entry(name)


@pytest.mark.test_set_ci
class TestRblnRuntimeLibResolve(TestCase):
    """Resolution order for librbln.so."""

    def _isolated_env(self, **overrides: str):
        """Blank the environment overrides so only the candidate under test can match."""
        env = dict.fromkeys(("LD_LIBRARY_PATH",), "")
        env.update(overrides)
        return patch.dict(os.environ, env, clear=False)

    def _make_root(self) -> str:
        """A throwaway directory to build a layout under."""
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        return root

    def _make_lib(self, *parts: str, root: str | None = None) -> str:
        """Create an empty file standing in for librbln.so and return its path.

        Pass ``root`` to place several files in one layout; without it each call gets its own,
        which silently defeats any test about one candidate outranking another.
        """
        path = os.path.join(root or self._make_root(), *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w"):
            pass
        return path

    def test_resolves_to_an_existing_library(self):
        path, source = rbln_runtime_lib.resolve_runtime_library()
        self.assertTrue(os.path.isfile(path), f"{path} (from {source}) does not exist")

    def test_ld_library_path_overrides_the_installed_library(self):
        # The only override, and the one the dynamic loader and rebel-compiler honour too, so all
        # three sides land on the same file.
        override = self._make_lib("custom", "librbln.so")
        with self._isolated_env(LD_LIBRARY_PATH=os.path.dirname(override)):
            path, source = rbln_runtime_lib.resolve_runtime_library()
        self.assertEqual(path, os.path.realpath(override))
        self.assertEqual(source, "LD_LIBRARY_PATH")

    def test_the_record_is_read_next_to_the_imported_package(self):
        # torch-rbln's own editable install is shaped like this: the distribution records artifacts
        # under site-packages while the imported package comes from the source tree. The two files
        # differ, and the copy beside the package this interpreter imports is the one in use.
        imported = self._make_lib("src", "custom", "librbln.so")
        installed = self._make_lib("site-packages", "custom", "librbln.so")
        anchor = os.path.dirname(os.path.dirname(imported))
        with (
            self._isolated_env(),
            patch.object(rbln_runtime_lib, "_rebel_package_anchor", return_value=anchor),
            patch.object(
                rbln_runtime_lib,
                "_record_entries",
                return_value=[_record("custom/librbln.so", installed)],
            ),
        ):
            path, source = rbln_runtime_lib.resolve_runtime_library()
        self.assertEqual(path, imported)
        self.assertIn("record", source)

    def test_the_record_follows_a_relocated_library(self):
        # The library moved out of the directory it has always been in; the record names it there
        # and nothing here needs changing for that.
        library = self._make_lib("somewhere", "new_home", "librbln.so")
        anchor = os.path.dirname(os.path.dirname(library))
        with (
            self._isolated_env(),
            patch.object(rbln_runtime_lib, "_rebel_package_anchor", return_value=anchor),
            patch.object(
                rbln_runtime_lib,
                "_record_entries",
                return_value=[_record("new_home/librbln.so", library)],
            ),
        ):
            path, source = rbln_runtime_lib.resolve_runtime_library()
        self.assertEqual(path, library)
        self.assertIn("record", source)

    def test_a_leftover_build_tree_does_not_outrank_the_record(self):
        # A source checkout that was built once keeps <src>/build/librbln.so around forever. It
        # answers only when the record does not, or it would shadow a distribution that names the
        # file exactly -- the state a relocating rebel-compiler leaves behind.
        root = self._make_root()
        stale = self._make_lib("build", "librbln.so", root=root)
        shipped = self._make_lib("site", "new_home", "librbln.so", root=root)
        anchor = os.path.join(root, "python")
        self.assertTrue(os.path.isfile(stale), "the leftover build tree must exist to compete")
        with (
            self._isolated_env(),
            patch.object(rbln_runtime_lib, "_rebel_package_anchor", return_value=anchor),
            patch.object(
                rbln_runtime_lib,
                "_record_entries",
                return_value=[_record("new_home/librbln.so", shipped)],
            ),
        ):
            path, _ = rbln_runtime_lib.resolve_runtime_library()
        self.assertEqual(path, shipped)

    def test_an_editable_install_falls_back_to_the_build_tree(self):
        # pip install -e <src>/python: the package is <src>/python/rebel, the library <src>/build,
        # and the distribution records neither -- which is why this tier exists at all.
        library = self._make_lib("build", "librbln.so")
        source_root = os.path.dirname(os.path.dirname(library))
        with (
            self._isolated_env(),
            patch.object(rbln_runtime_lib, "_rebel_package_anchor", return_value=os.path.join(source_root, "python")),
            patch.object(rbln_runtime_lib, "_record_entries", return_value=[]),
        ):
            path, source = rbln_runtime_lib.resolve_runtime_library()
        self.assertEqual(path, library)
        self.assertEqual(source, "rebel source build tree")

    def test_missing_library_reports_what_was_tried(self):
        with (
            self._isolated_env(),
            patch.object(rbln_runtime_lib, "_rebel_package_anchor", return_value="/nonexistent/anchor"),
            patch.object(rbln_runtime_lib, "_record_entries", return_value=[]),
        ):
            with self.assertRaises(FileNotFoundError) as ctx:
                rbln_runtime_lib.resolve_runtime_library()
        message = str(ctx.exception)
        self.assertIn("Could not find librbln.so", message)
        self.assertIn("/nonexistent/build/librbln.so (from rebel source build tree)", message)
        self.assertIn("LD_LIBRARY_PATH", message)
        self.assertIn("python -m torch_rbln.diagnose", message)

    def test_empty_path_entries_are_dropped(self):
        # A doubled separator used to resolve to the current working directory.
        with patch.dict(os.environ, {"LD_LIBRARY_PATH": "/a::/b:"}, clear=False):
            self.assertEqual(rbln_runtime_lib._split_env_paths("LD_LIBRARY_PATH"), ["/a", "/b"])


@pytest.mark.test_set_ci
class TestRblnRuntimeLibRecord(TestCase):
    """What the build takes from the same record."""

    def test_the_install_relative_directory_comes_from_the_record(self):
        # The install RPATH is written against the directory the package sits in, which a recorded
        # path already expresses -- no arithmetic against whichever site-packages the build ran in.
        library, _ = rbln_runtime_lib.resolve_runtime_library()
        self.assertEqual(rbln_runtime_lib.record_relative_dir(library), os.path.basename(os.path.dirname(library)))

    def test_a_library_outside_the_record_has_no_relative_directory(self):
        self.assertIsNone(rbln_runtime_lib.record_relative_dir("/nowhere/librbln.so"))

    def test_the_include_directory_comes_from_a_recorded_header(self):
        include_dir = rbln_runtime_lib.record_include_dir()
        self.assertIsNotNone(include_dir)
        self.assertTrue(os.path.isdir(include_dir), f"{include_dir} does not exist")


instantiate_device_type_tests(TestRblnRuntimeLibResolve, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestRblnRuntimeLibRecord, globals(), only_for="privateuse1")

if __name__ == "__main__":
    run_tests()
