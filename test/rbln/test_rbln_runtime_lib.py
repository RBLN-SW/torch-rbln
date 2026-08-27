# Owner(s): ["module: PrivateUse1"]

"""Tests for how torch_rbln._internal.rbln_runtime_lib locates librbln.so.

The rule under test is that the rebel-compiler distribution's record of what it installed
decides, with LD_LIBRARY_PATH as the one override, and that nothing guesses at a layout ahead
of that record.
"""

import os
import shutil
import sys
import tempfile
import types
import unittest
from importlib.metadata import distribution, PackageNotFoundError
from unittest.mock import patch

import pytest
from torch.testing._internal.common_utils import run_tests, TestCase

from torch_rbln._internal import rbln_runtime_lib


def _compiler_is_installed() -> bool:
    """Whether this environment has a rebel-compiler distribution for the resolver to find.

    The rest of this file synthesizes what it needs, so it runs on a host with no compiler and no
    device. Only the checks against the real install depend on one.
    """
    try:
        distribution("rebel-compiler")
    except PackageNotFoundError:
        return False
    return True


_NEEDS_COMPILER = unittest.skipUnless(_compiler_is_installed(), "needs an installed rebel-compiler")


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

    @_NEEDS_COMPILER
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
            patch.object(rbln_runtime_lib, "_rebel_package_anchors", return_value=[anchor]),
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
            patch.object(rbln_runtime_lib, "_rebel_package_anchors", return_value=[anchor]),
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
        # A source checkout that was built once keeps <src>/build/librbln.so around forever, so it
        # answers only when the record does not -- otherwise it shadows a distribution that names
        # the file exactly. The distribution installed the package in use here, so its record
        # speaks for it.
        root = self._make_root()
        stale = self._make_lib("build", "librbln.so", root=root)
        shipped = self._make_lib("site", "new_home", "librbln.so", root=root)
        anchor = os.path.join(root, "site")
        self.assertTrue(os.path.isfile(stale), "the leftover build tree must exist to compete")
        with (
            self._isolated_env(),
            patch.object(rbln_runtime_lib, "_rebel_package_anchors", return_value=[anchor]),
            patch.object(
                rbln_runtime_lib,
                "_record_entries",
                return_value=[_record("new_home/librbln.so", shipped)],
            ),
        ):
            path, _ = rbln_runtime_lib.resolve_runtime_library()
        self.assertEqual(path, shipped)

    def test_a_record_from_another_tree_does_not_answer_for_the_imported_package(self):
        # A wheel installed in site-packages while a checkout on PYTHONPATH provides the package:
        # the recorded library belongs to the wheel's Python, and rebel-compiler loads the one
        # beside its own files -- the checkout's. Taking the recorded one maps both.
        root = self._make_root()
        checkout_build = self._make_lib("checkout", "build", "librbln.so", root=root)
        installed = self._make_lib("site-packages", "shipped", "librbln.so", root=root)
        anchor = os.path.join(root, "checkout", "python")
        self.assertTrue(os.path.isfile(installed), "the installed library must exist to compete")
        with (
            self._isolated_env(),
            patch.object(rbln_runtime_lib, "_rebel_package_anchors", return_value=[anchor]),
            patch.object(
                rbln_runtime_lib,
                "_record_entries",
                return_value=[_record("shipped/librbln.so", installed)],
            ),
        ):
            path, source = rbln_runtime_lib.resolve_runtime_library()
        self.assertEqual(path, checkout_build)
        self.assertEqual(source, "rebel source build tree")

    def test_an_editable_install_falls_back_to_the_build_tree(self):
        # pip install -e <src>/python: the package is <src>/python/rebel, the library <src>/build,
        # and the distribution records neither.
        library = self._make_lib("build", "librbln.so")
        source_root = os.path.dirname(os.path.dirname(library))
        with (
            self._isolated_env(),
            patch.object(
                rbln_runtime_lib, "_rebel_package_anchors", return_value=[os.path.join(source_root, "python")]
            ),
            patch.object(rbln_runtime_lib, "_record_entries", return_value=[]),
        ):
            path, source = rbln_runtime_lib.resolve_runtime_library()
        self.assertEqual(path, library)
        self.assertEqual(source, "rebel source build tree")

    def test_the_origin_decides_which_tree_provides_the_library(self):
        # An editable install reports its search locations out of a set holding both the source
        # tree and the build tree, so their order changes from process to process. ``origin`` names
        # the __init__.py that will run: one value, and the tree rebel-compiler loads beside.
        # Both trees hold a library, and the one that does not provide rebel sorts first, so an
        # order-based choice picks the wrong file rather than merely trying it.
        root = self._make_root()
        library = self._make_lib("checkout", "build", "librbln.so", root=root)
        elsewhere = self._make_lib("another", "build", "librbln.so", root=root)
        package = os.path.join(root, "checkout", "python", "rebel")
        staging = os.path.join(root, "another", "python", "rebel")
        self.assertTrue(os.path.isfile(elsewhere), "the other tree's library must exist to compete")
        for locations in ([package, staging], [staging, package]):
            spec = types.SimpleNamespace(
                origin=os.path.join(package, "__init__.py"), submodule_search_locations=locations
            )
            with (
                self._isolated_env(),
                patch.object(rbln_runtime_lib.importlib.util, "find_spec", return_value=spec),
                patch.object(rbln_runtime_lib, "_record_entries", return_value=[]),
            ):
                path, source = rbln_runtime_lib.resolve_runtime_library()
            self.assertEqual(path, library, f"locations in the order {locations} resolved elsewhere")
            self.assertEqual(source, "rebel source build tree")

    def test_the_search_locations_answer_when_there_is_no_origin(self):
        # A directory an uninstall left behind imports as a namespace package: no origin, and the
        # search locations are all there is to go on.
        root = self._make_root()
        library = self._make_lib("checkout", "build", "librbln.so", root=root)
        spec = types.SimpleNamespace(
            origin=None, submodule_search_locations=[os.path.join(root, "checkout", "python", "rebel")]
        )
        with (
            self._isolated_env(),
            patch.object(rbln_runtime_lib.importlib.util, "find_spec", return_value=spec),
            patch.object(rbln_runtime_lib, "_record_entries", return_value=[]),
        ):
            path, source = rbln_runtime_lib.resolve_runtime_library()
        self.assertEqual(path, library)
        self.assertEqual(source, "rebel source build tree")

    def test_missing_library_reports_what_was_tried(self):
        with (
            self._isolated_env(),
            patch.object(rbln_runtime_lib, "_rebel_package_anchors", return_value=["/nonexistent/anchor"]),
            patch.object(rbln_runtime_lib, "_record_entries", return_value=[]),
        ):
            with self.assertRaises(FileNotFoundError) as ctx:
                rbln_runtime_lib.resolve_runtime_library()
        message = str(ctx.exception)
        self.assertIn("Could not find librbln.so", message)
        self.assertIn("/nonexistent/build/librbln.so (from rebel source build tree)", message)
        self.assertIn("LD_LIBRARY_PATH", message)
        self.assertIn("python -m torch_rbln.diagnose", message)

    def test_an_absent_distribution_is_named_as_the_reason(self):
        # With nothing installed there is no candidate to report, and a bare list of paths reads as
        # a path problem. The message has to name the interpreter that is missing the package.
        with (
            self._isolated_env(),
            patch.object(rbln_runtime_lib, "distribution", side_effect=PackageNotFoundError("rebel-compiler")),
            patch.object(rbln_runtime_lib, "_rebel_package_anchors", return_value=["/site-packages"]),
        ):
            with self.assertRaises(FileNotFoundError) as ctx:
                rbln_runtime_lib.resolve_runtime_library()
        message = str(ctx.exception)
        self.assertIn("No rebel-compiler distribution is installed for", message)
        self.assertIn(sys.executable, message)
        self.assertIn("/site-packages", message)

    def test_empty_path_entries_are_dropped(self):
        # A doubled separator used to resolve to the current working directory.
        with patch.dict(os.environ, {"LD_LIBRARY_PATH": "/a::/b:"}, clear=False):
            self.assertEqual(rbln_runtime_lib._split_env_paths("LD_LIBRARY_PATH"), ["/a", "/b"])


@pytest.mark.test_set_ci
class TestRblnRuntimeLibRecord(TestCase):
    """What the build takes from the same record."""

    def test_the_install_relative_directory_is_the_whole_recorded_path(self):
        # Whatever the record says, not a component of it: rebel-compiler's target layout nests the
        # library one level deeper (rebel/lib).
        library = "/fake/site/rebel/lib/librbln.so"
        with (
            patch.object(rbln_runtime_lib, "_rebel_package_anchors", return_value=["/fake/site"]),
            patch.object(
                rbln_runtime_lib,
                "_record_entries",
                return_value=[_record("rebel/lib/librbln.so", library)],
            ),
        ):
            self.assertEqual(rbln_runtime_lib.record_relative_dir(library), os.path.join("rebel", "lib"))

    @_NEEDS_COMPILER
    def test_the_installed_library_is_under_its_recorded_directory(self):
        # The same rule against the real install, at whatever depth it happens to sit.
        library, _ = rbln_runtime_lib.resolve_runtime_library()
        reldir = rbln_runtime_lib.record_relative_dir(library)
        self.assertIsNotNone(reldir)
        self.assertTrue(
            os.path.realpath(library).endswith(os.path.join(reldir, os.path.basename(library))),
            f"{library} does not sit under the recorded {reldir}",
        )

    def test_a_library_outside_the_record_has_no_relative_directory(self):
        self.assertIsNone(rbln_runtime_lib.record_relative_dir("/nowhere/librbln.so"))

    @_NEEDS_COMPILER
    def test_the_include_directory_comes_from_a_recorded_header(self):
        include_dir = rbln_runtime_lib.record_include_dir()
        self.assertIsNotNone(include_dir)
        self.assertTrue(os.path.isdir(include_dir), f"{include_dir} does not exist")


@pytest.mark.test_set_ci
class TestRblnRuntimeLibAdopt(TestCase):
    """Adopting a librbln.so something else mapped first."""

    def test_an_existing_mapping_is_adopted_without_loading_again(self):
        with (
            patch.object(rbln_runtime_lib, "loaded_runtime_libraries", return_value=["/opt/rebel/librbln.so"]),
            patch.object(rbln_runtime_lib.ctypes, "CDLL", side_effect=AssertionError("loaded again")),
        ):
            self.assertEqual(rbln_runtime_lib.load_runtime_library(), "/opt/rebel/librbln.so")

    def test_a_deleted_mapping_is_reported_as_a_path(self):
        # /proc/self/maps marks a mapping whose file is gone with " (deleted)". The marker is
        # not part of the path, and what this returns is handed to dlopen by the ABI check.
        with patch.object(
            rbln_runtime_lib, "loaded_runtime_libraries", return_value=["/opt/rebel/librbln.so (deleted)"]
        ):
            self.assertEqual(rbln_runtime_lib.load_runtime_library(), "/opt/rebel/librbln.so")


if __name__ == "__main__":
    run_tests()
