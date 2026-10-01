# Owner(s): ["module: PrivateUse1"]

"""
Test the rbln ABI check performed when torch_rbln loads the rbln runtime.

Covers the equality verdict and its report, the cases that fail open (no handle on the mapped
runtime, no build-time id, a runtime without ``rbln_abi_id``), the opt-out env var, how the
snapshot is read, that the build's snapshot template yields a readable id, and that the
runtime installed here agrees with this build.
"""

import contextlib
import hashlib
import os
import shutil
import subprocess
import tempfile
import types


# Keeps torch_backends_entry_point from running during this import; restored right after so it
# does not leak into other test modules on the same xdist worker (env_utils reads it live).
_prev_diagnose = os.environ.get("TORCH_RBLN_DIAGNOSE")
os.environ["TORCH_RBLN_DIAGNOSE"] = "1"
try:
    from unittest.mock import patch

    import pytest
    from torch.testing._internal.common_utils import run_tests, TestCase

    import torch_rbln  # noqa: F401  -- gated import, keeps the backend from initialising here
    from torch_rbln._internal import abi_check, rbln_runtime_lib
finally:
    # Restore in a finally so a failing gated import cannot leak DIAGNOSE onto the worker.
    if _prev_diagnose is None:
        os.environ.pop("TORCH_RBLN_DIAGNOSE", None)
    else:
        os.environ["TORCH_RBLN_DIAGNOSE"] = _prev_diagnose


# An empty value reads as "not disabled", so this pins the opt-out off regardless of
# what the machine running the tests has exported.
_ABI_CHECK_ENABLED = {abi_check._SKIP_ENV: ""}

_SNAPSHOT_MODULE = "torch_rbln._internal._abi_snapshot"
_FAKE_PATH = "/fake/rbln/build/rbln/librbln_rt.so"
_ID_A = "a" * 64
_ID_B = "b" * 64


class _FakeFunc:
    """A resolved symbol: ctypes lets the caller set restype/argtypes, then call it."""

    def __init__(self, value: bytes | None) -> None:
        self._value = value
        self.restype = None
        self.argtypes = None

    def __call__(self) -> bytes | None:
        return self._value


class _FakeLib:
    """Stands in for a ctypes.CDLL, where attribute access is a dlsym lookup."""

    def __init__(self, symbols: dict) -> None:
        self._symbols = symbols

    def __getattr__(self, item: str) -> "_FakeFunc":
        try:
            return _FakeFunc(self._symbols[item])
        except KeyError:
            # Mirrors ctypes on Linux: an unresolvable symbol raises AttributeError.
            raise AttributeError(f"{_FAKE_PATH}: undefined symbol: {item}") from None


def _runtime_reporting(abi_id: str) -> _FakeLib:
    return _FakeLib({abi_check.ABI_SYMBOL: abi_id.encode()})


def _snapshot(**values) -> types.ModuleType:
    module = types.ModuleType(_SNAPSHOT_MODULE)
    module.__dict__.update(values)
    return module


@contextlib.contextmanager
def _runtime(lib, built_abi, built_include_dir="/fake/rbln/rbln/include"):
    """The check against a fake runtime and a fixed snapshot, opt-out pinned off."""
    with (
        patch.dict(os.environ, _ABI_CHECK_ENABLED),
        patch.dict(
            "sys.modules", {_SNAPSHOT_MODULE: _snapshot(BUILT_ABI=built_abi, BUILT_INCLUDE_DIR=built_include_dir)}
        ),
        patch.object(abi_check, "open_mapped_runtime", return_value=lib),
    ):
        yield


@pytest.mark.test_set_ci
class TestSnapshot(TestCase):
    """The build-time snapshot, including the shapes that mean 'no id recorded'."""

    def test_missing_generated_module_reads_as_no_snapshot(self):
        # A None entry in sys.modules makes the import raise ImportError.
        with patch.dict("sys.modules", {_SNAPSHOT_MODULE: None}):
            self.assertIsNone(abi_check.get_built_abi())
            self.assertIsNone(abi_check.get_built_include_dir())

    def test_values_that_are_not_a_sha256_hex_digest_read_as_no_snapshot(self):
        for value in (None, "", "abc", "A" * 64, "a" * 63, "a" * 65, "g" * 64, 1, b"a" * 64):
            with patch.dict("sys.modules", {_SNAPSHOT_MODULE: _snapshot(BUILT_ABI=value)}):
                self.assertIsNone(abi_check.get_built_abi(), f"value={value!r}")

    def test_a_sha256_hex_digest_is_the_snapshot(self):
        with patch.dict("sys.modules", {_SNAPSHOT_MODULE: _snapshot(BUILT_ABI=_ID_A)}):
            self.assertEqual(abi_check.get_built_abi(), _ID_A)

    def test_include_dir_is_read_when_recorded(self):
        with patch.dict("sys.modules", {_SNAPSHOT_MODULE: _snapshot(BUILT_INCLUDE_DIR="/tree/rbln/include")}):
            self.assertEqual(abi_check.get_built_include_dir(), "/tree/rbln/include")
        for value in ("", None):
            with patch.dict("sys.modules", {_SNAPSHOT_MODULE: _snapshot(BUILT_INCLUDE_DIR=value)}):
                self.assertIsNone(abi_check.get_built_include_dir(), f"value={value!r}")


@pytest.mark.test_set_ci
class TestReadRuntimeAbi(TestCase):
    """Reading ``rbln_abi_id`` off a loaded runtime."""

    def test_reads_the_id(self):
        self.assertEqual(abi_check.read_runtime_abi(_runtime_reporting(_ID_A)), _ID_A)

    def test_a_runtime_without_the_symbol_reports_none(self):
        self.assertIsNone(abi_check.read_runtime_abi(_FakeLib({})))

    def test_a_null_id_reports_none(self):
        self.assertIsNone(abi_check.read_runtime_abi(_FakeLib({abi_check.ABI_SYMBOL: None})))

    def test_no_handle_reports_none(self):
        self.assertIsNone(abi_check.read_runtime_abi(None))


@pytest.mark.test_set_ci
class TestOpenMappedRuntime(TestCase):
    """Taking a handle on the library the import path already mapped."""

    def test_a_path_that_cannot_be_opened_is_not_an_exception(self):
        # /proc/self/maps can name a mapping whose file has since been replaced; this is reached
        # from the import path, where an OSError would kill the import.
        self.assertIsNone(abi_check.open_mapped_runtime("/nonexistent/librbln_rt.so"))

    def test_only_a_mapped_library_gives_a_handle(self):
        # RTLD_NOLOAD: a copy of a mapped library is another file, so it yields no handle
        # rather than being loaded next to the original.
        with open("/proc/self/maps") as maps:
            mapped = rbln_runtime_lib.parse_mapped_libraries(maps, name="libc.so.6")
        if not mapped:
            self.skipTest("libc.so.6 is not mapped under that name")
        self.assertIsNotNone(abi_check.open_mapped_runtime(mapped[0]))
        with tempfile.TemporaryDirectory() as tmp:
            copy = shutil.copy(mapped[0], tmp)
            self.assertIsNone(abi_check.open_mapped_runtime(copy))

    def test_no_path_gives_no_handle(self):
        self.assertIsNone(abi_check.open_mapped_runtime(None))
        self.assertIsNone(abi_check.open_mapped_runtime(""))


@pytest.mark.test_set_ci
class TestCheckRuntimeAbi(TestCase):
    """End-to-end verdicts, including what does and does not block the import."""

    def test_equal_ids_pass(self):
        with _runtime(_runtime_reporting(_ID_A), built_abi=_ID_A):
            self.assertEqual(abi_check.check_runtime_abi(_FAKE_PATH), abi_check.VERDICT_OK)

    def test_different_ids_raise_with_an_actionable_report(self):
        with _runtime(_runtime_reporting(_ID_B), built_abi=_ID_A, built_include_dir="/old/tree/rbln/include"):
            with self.assertRaises(ImportError) as ctx:
                abi_check.check_runtime_abi(_FAKE_PATH)
        message = str(ctx.exception)
        self.assertIn("RBLN ABI mismatch", message)
        self.assertIn(_FAKE_PATH, message)
        self.assertIn(f"runtime ABI:  {_ID_B}", message)
        self.assertIn(f"built ABI:    {_ID_A}", message)
        self.assertIn("/old/tree/rbln/include", message)
        self.assertIn("Rebuild torch-rbln with REBEL_HOME", message)
        self.assertIn("python -m torch_rbln.diagnose", message)

    def test_unreadable_runtime_warns_but_does_not_block(self):
        with _runtime(None, built_abi=_ID_A):
            with pytest.warns(UserWarning, match="no handle could be taken"):
                verdict = abi_check.check_runtime_abi(_FAKE_PATH)
        self.assertEqual(verdict, abi_check.VERDICT_SKIPPED_UNREADABLE_RUNTIME)

    def test_missing_snapshot_warns_but_does_not_block(self):
        with _runtime(_runtime_reporting(_ID_A), built_abi=None):
            with pytest.warns(UserWarning, match="recorded no rbln ABI id"):
                verdict = abi_check.check_runtime_abi(_FAKE_PATH)
        self.assertEqual(verdict, abi_check.VERDICT_SKIPPED_NO_SNAPSHOT)

    def test_runtime_without_an_id_warns_but_does_not_block(self):
        with _runtime(_FakeLib({}), built_abi=_ID_A):
            with pytest.warns(UserWarning, match="exports no rbln_abi_id"):
                verdict = abi_check.check_runtime_abi(_FAKE_PATH)
        self.assertEqual(verdict, abi_check.VERDICT_SKIPPED_NO_RUNTIME_ID)

    def test_no_snapshot_and_no_runtime_id_warn_once(self):
        # Both fail open, so the one this build can act on wins rather than stacking two warnings.
        with _runtime(_FakeLib({}), built_abi=None):
            with pytest.warns(UserWarning) as record:
                verdict = abi_check.check_runtime_abi(_FAKE_PATH)
        self.assertEqual(verdict, abi_check.VERDICT_SKIPPED_NO_SNAPSHOT)
        self.assertEqual(len(record), 1, [str(w.message) for w in record])

    def test_inspection_reports_both_ids_whatever_the_verdict(self):
        # diagnose shows the runtime's id even when this build has none to compare it with.
        with _runtime(_runtime_reporting(_ID_B), built_abi=None):
            state = abi_check.inspect_runtime_abi(_FAKE_PATH)
        self.assertEqual(state.verdict, abi_check.VERDICT_SKIPPED_NO_SNAPSHOT)
        self.assertEqual(state.runtime_abi, _ID_B)
        self.assertEqual(state.runtime_path, _FAKE_PATH)

    def test_opt_out_skips_a_mismatch_that_would_otherwise_raise(self):
        for value in ("1", "ON", "on", "true", "yes"):
            with (
                patch.dict(os.environ, {abi_check._SKIP_ENV: value}),
                patch.dict("sys.modules", {_SNAPSHOT_MODULE: _snapshot(BUILT_ABI=_ID_A)}),
                # The opt-out has to cover every step, taking the handle included, or it
                # cannot unblock a machine where that step is what fails.
                patch.object(abi_check, "open_mapped_runtime", side_effect=AssertionError("opened")),
            ):
                self.assertEqual(
                    abi_check.check_runtime_abi(_FAKE_PATH),
                    abi_check.VERDICT_SKIPPED_DISABLED,
                    f"value={value!r}",
                )

    def test_opt_out_ignores_empty_and_off_values(self):
        for value in ("", "0", "off", "  "):
            with patch.dict(os.environ, {abi_check._SKIP_ENV: value}):
                self.assertFalse(abi_check.is_abi_check_disabled(), f"value={value!r}")


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TEMPLATE = os.path.join(_REPO_ROOT, "cmake", "abi_snapshot.py.in")


@pytest.mark.test_set_ci
class TestSnapshotTemplate(TestCase):
    """The snapshot the build generates, from the runtime's own script and this repo's template."""

    def test_the_generated_snapshot_records_the_id_of_the_headers(self):
        rebel_home = os.environ.get("REBEL_HOME")
        if not rebel_home:
            self.skipTest("REBEL_HOME is not set")
        script = os.path.join(rebel_home, "rbln", "cmake", "RblnAbi.cmake")
        include_dir = os.path.join(rebel_home, "rbln", "include")
        cmake = shutil.which("cmake")
        if cmake is None or not os.path.isfile(script):
            self.skipTest("needs cmake and a REBEL_HOME tree with rbln/cmake/RblnAbi.cmake")

        with tempfile.TemporaryDirectory() as tmp:
            output = os.path.join(tmp, "_abi_snapshot.py")
            subprocess.run(
                [cmake, f"-DINCLUDE_DIR={include_dir}", f"-DTEMPLATE={_TEMPLATE}", f"-DOUTPUT={output}", "-P", script],
                check=True,
                capture_output=True,
            )
            namespace: dict = {}
            with open(output) as f:
                exec(f.read(), namespace)

        # The id's definition, restated: SHA-256 over "<relative path> <SHA-256>\n" of every
        # rbln/**/*.h, sorted by path.
        headers = sorted(
            os.path.relpath(os.path.join(root, name), include_dir)
            for root, _, names in os.walk(os.path.join(include_dir, "rbln"))
            for name in names
            if name.endswith(".h")
        )
        text = "".join(
            f"{header} {hashlib.sha256(open(os.path.join(include_dir, header), 'rb').read()).hexdigest()}\n"
            for header in headers
        )
        expected = hashlib.sha256(text.encode()).hexdigest()

        with patch.dict("sys.modules", {_SNAPSHOT_MODULE: _snapshot(**namespace)}):
            self.assertEqual(abi_check.get_built_abi(), expected)
            self.assertEqual(abi_check.get_built_include_dir(), include_dir)


def _mapped_runtime(case) -> str:
    """The runtime the import path maps, or skip if rbln is not importable here."""
    try:
        return rbln_runtime_lib.load_runtime_library()
    except ImportError as e:
        case.skipTest(f"the rbln runtime is not available: {e}")


@pytest.mark.test_set_ci
class TestInstalledCombination(TestCase):
    """The runtime importable on this machine against the torch-rbln under test."""

    def test_the_mapped_runtime_reports_an_id(self):
        # The check reads the id off this handle; a None would silently turn every check into
        # the fail-open path.
        lib = abi_check.open_mapped_runtime(_mapped_runtime(self))
        self.assertIsNotNone(lib)
        self.assertRegex(abi_check.read_runtime_abi(lib) or "", r"^[0-9a-f]{64}$")

    def test_this_build_and_this_runtime_agree(self):
        state = abi_check.inspect_runtime_abi(_mapped_runtime(self))
        if state.built_abi is None:
            self.skipTest("this torch-rbln recorded no ABI id (not built through CMake)")
        self.assertEqual(
            state.verdict,
            abi_check.VERDICT_OK,
            f"runtime {state.runtime_path} reports {state.runtime_abi}, this build recorded {state.built_abi}",
        )


if __name__ == "__main__":
    run_tests()
