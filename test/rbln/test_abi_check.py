# Owner(s): ["module: PrivateUse1"]

"""
Test the rebel ABI handshake performed when torch_rbln loads librbln.so.

Covers the accept/reject window (min_supported <= built <= current), the cases that
fail open (pre-handshake runtime, half-exported runtime, no handle on the mapped
library, no build-time snapshot), the ones that fail closed with no snapshot at all,
the opt-out env var, and that the librbln.so installed here agrees with this build.
"""

import contextlib
import os


# Gate torch_backends_entry_point during our import of torch_rbln (it would otherwise load
# librbln.so). Restore the flag right after the import so it does not leak into other test
# modules collected on the same xdist worker (env_utils reads TORCH_RBLN_DIAGNOSE live).
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


class _FakeFunc:
    """A resolved symbol: ctypes lets the caller set restype/argtypes, then call it."""

    def __init__(self, value: int) -> None:
        self._value = value
        self.restype = None
        self.argtypes = None

    def __call__(self) -> int:
        return self._value


class _FakeLib:
    """Stands in for a ctypes.CDLL, where attribute access is a dlsym lookup."""

    def __init__(self, symbols: dict, name: str = "/fake/tvm/librbln.so") -> None:
        self._symbols = symbols
        self._name = name

    def __getattr__(self, item: str) -> "_FakeFunc":
        try:
            return _FakeFunc(self._symbols[item])
        except KeyError:
            # Mirrors ctypes on Linux: an unresolvable symbol raises AttributeError.
            raise AttributeError(f"{self._name}: undefined symbol: {item}") from None


def _lib_with_window(min_supported: int, current: int) -> _FakeLib:
    return _FakeLib({"rbln_abi_min_supported": min_supported, "rbln_abi_current": current})


_FAKE_PATH = "/fake/tvm/librbln.so"


@contextlib.contextmanager
def _runtime(lib, built_abi):
    """check_librbln_abi against a fake librbln.so and a fixed snapshot, opt-out pinned off."""
    with (
        patch.dict(os.environ, _ABI_CHECK_ENABLED),
        patch.object(abi_check, "get_built_abi", return_value=built_abi),
        patch.object(abi_check, "open_mapped_runtime", return_value=lib),
    ):
        yield


@pytest.mark.test_set_ci
class TestAbiMismatchReason(TestCase):
    """The accept/reject window, for every combination of consumer and runtime."""

    def test_exact_match_is_accepted(self):
        self.assertIsNone(abi_check.abi_mismatch_reason(1, 1, 1))

    def test_additive_runtime_still_accepts_older_consumer(self):
        # v2 added API but kept min at 1: consumers built at 1 and 2 both pass.
        self.assertIsNone(abi_check.abi_mismatch_reason(1, 1, 2))
        self.assertIsNone(abi_check.abi_mismatch_reason(2, 1, 2))

    def test_consumer_newer_than_runtime_is_rejected(self):
        # The runtime is the old side, so the fix points at rebel-compiler.
        reason = abi_check.abi_mismatch_reason(2, 1, 1)
        self.assertIsNotNone(reason)
        self.assertIn("Upgrade rebel-compiler", reason)

    def test_consumer_below_min_supported_is_rejected(self):
        # v3 was breaking (min = current = 3), so consumers at 1 and 2 are cut off and
        # the fix points the other way, at torch-rbln.
        for built in (1, 2):
            reason = abi_check.abi_mismatch_reason(built, 3, 3)
            self.assertIsNotNone(reason)
            self.assertIn("no longer accepts consumers below ABI 3", reason)
            self.assertIn("Install a torch-rbln built against this rebel-compiler", reason)

    def test_reason_names_both_numbers(self):
        reason = abi_check.abi_mismatch_reason(7, 3, 3)
        self.assertIn("7", reason)
        self.assertIn("3", reason)

    def test_inconsistent_runtime_window_is_rejected(self):
        # min > current cannot describe any acceptable consumer, so nothing passes.
        reason = abi_check.abi_mismatch_reason(4, 5, 4)
        self.assertIsNotNone(reason)
        self.assertIn("inconsistent", reason)


@pytest.mark.test_set_ci
class TestReadRuntimeAbi(TestCase):
    """Reading the two entry points off a loaded librbln.so."""

    def test_reads_min_then_current(self):
        self.assertEqual(abi_check.read_runtime_abi(_lib_with_window(1, 4)), (1, 4))

    def test_missing_both_symbols_is_pre_abi(self):
        self.assertIsNone(abi_check.read_runtime_abi(_FakeLib({})))

    def test_half_exported_runtime_has_no_window(self):
        # A .so with only one of the pair is malformed, but there is still no window to
        # validate against, so read_runtime_abi reports none either way.
        self.assertIsNone(abi_check.read_runtime_abi(_FakeLib({"rbln_abi_current": 2})))
        self.assertIsNone(abi_check.read_runtime_abi(_FakeLib({"rbln_abi_min_supported": 2})))

    def test_no_handle_is_pre_abi(self):
        self.assertIsNone(abi_check.read_runtime_abi(None))

    def test_symbols_report_which_names_were_missing(self):
        # What separates a runtime that predates the handshake from a malformed one.
        self.assertEqual(abi_check.read_abi_symbols(_lib_with_window(1, 4)), ([1, 4], []))
        self.assertEqual(abi_check.read_abi_symbols(_FakeLib({})), ([], list(abi_check.ABI_SYMBOLS)))
        self.assertEqual(
            abi_check.read_abi_symbols(_FakeLib({"rbln_abi_current": 2})),
            ([2], ["rbln_abi_min_supported"]),
        )
        self.assertEqual(abi_check.read_abi_symbols(None), ([], list(abi_check.ABI_SYMBOLS)))


@pytest.mark.test_set_ci
class TestOpenMappedRuntime(TestCase):
    """Taking a handle on the library the import path already mapped."""

    def test_a_path_that_cannot_be_opened_is_not_an_exception(self):
        # /proc/self/maps can name a mapping whose file has since been replaced. The handle
        # is what the check needs, not what the process runs on, so this must not raise --
        # it is reached from the import path, where an OSError would kill the import.
        self.assertIsNone(abi_check.open_mapped_runtime("/nonexistent/librbln.so"))
        self.assertIsNone(abi_check.open_mapped_runtime("/nonexistent/librbln.so (deleted)"))

    def test_no_path_gives_no_handle(self):
        self.assertIsNone(abi_check.open_mapped_runtime(None))
        self.assertIsNone(abi_check.open_mapped_runtime(""))


@pytest.mark.test_set_ci
class TestGetBuiltAbi(TestCase):
    """The build-time snapshot, including the shapes that mean 'no snapshot'."""

    def test_missing_generated_module_reads_as_no_snapshot(self):
        with patch.dict("sys.modules", {"torch_rbln._internal._abi_snapshot": None}):
            # A None entry in sys.modules makes the import raise ImportError.
            self.assertIsNone(abi_check.get_built_abi())

    def test_none_and_non_positive_and_bool_read_as_no_snapshot(self):
        for value in (None, 0, -1, True, "1", 1.0):
            fake = type("M", (), {"BUILT_ABI": value})
            with patch.dict("sys.modules", {"torch_rbln._internal._abi_snapshot": fake}):
                self.assertIsNone(abi_check.get_built_abi(), f"value={value!r}")

    def test_positive_int_is_the_snapshot(self):
        fake = type("M", (), {"BUILT_ABI": 3})
        with patch.dict("sys.modules", {"torch_rbln._internal._abi_snapshot": fake}):
            self.assertEqual(abi_check.get_built_abi(), 3)


@pytest.mark.test_set_ci
class TestCheckLibrblnAbi(TestCase):
    """End-to-end verdicts, including what does and does not block the import."""

    def test_in_window_passes(self):
        with _runtime(_lib_with_window(1, 3), built_abi=2):
            self.assertEqual(abi_check.check_librbln_abi(_FAKE_PATH), abi_check.VERDICT_OK)

    def test_out_of_window_raises_with_an_actionable_report(self):
        with _runtime(_lib_with_window(1, 3), built_abi=5):
            with self.assertRaises(ImportError) as ctx:
                abi_check.check_librbln_abi(_FAKE_PATH)
        message = str(ctx.exception)
        self.assertIn("RBLN ABI mismatch", message)
        self.assertIn("/fake/tvm/librbln.so", message)
        self.assertIn("built against rebel ABI 5", message)
        self.assertIn("python -m torch_rbln.diagnose", message)

    def test_pre_abi_runtime_warns_but_does_not_block(self):
        with _runtime(_FakeLib({}), built_abi=2):
            with pytest.warns(UserWarning, match="no ABI version symbols"):
                verdict = abi_check.check_librbln_abi(_FAKE_PATH)
        self.assertEqual(verdict, abi_check.VERDICT_SKIPPED_PRE_ABI_RUNTIME)

    def test_half_exported_runtime_warns_as_malformed_but_does_not_block(self):
        # Distinct from the pre-ABI case: the pair ships together, so one of them alone is a
        # broken runtime, not an old one, and the warning has to say so.
        with _runtime(_FakeLib({"rbln_abi_current": 2}), built_abi=2):
            with pytest.warns(UserWarning, match="malformed rather than merely old"):
                verdict = abi_check.check_librbln_abi(_FAKE_PATH)
        self.assertEqual(verdict, abi_check.VERDICT_SKIPPED_MALFORMED_RUNTIME)

    def test_unreadable_runtime_warns_but_does_not_block(self):
        with _runtime(None, built_abi=2):
            with pytest.warns(UserWarning, match="no handle could be taken"):
                verdict = abi_check.check_librbln_abi(_FAKE_PATH)
        self.assertEqual(verdict, abi_check.VERDICT_SKIPPED_UNREADABLE_RUNTIME)

    def test_missing_snapshot_warns_but_does_not_block(self):
        with _runtime(_lib_with_window(1, 1), built_abi=None):
            with pytest.warns(UserWarning, match="recorded no rebel ABI number"):
                verdict = abi_check.check_librbln_abi(_FAKE_PATH)
        self.assertEqual(verdict, abi_check.VERDICT_SKIPPED_NO_SNAPSHOT)

    def test_a_pre_abi_runtime_and_no_snapshot_warn_once(self):
        # Both fail open, so the actionable one wins rather than stacking two warnings.
        with _runtime(_FakeLib({}), built_abi=None):
            with pytest.warns(UserWarning) as record:
                verdict = abi_check.check_librbln_abi(_FAKE_PATH)
        self.assertEqual(verdict, abi_check.VERDICT_SKIPPED_NO_SNAPSHOT)
        self.assertEqual(len(record), 1, [str(w.message) for w in record])

    def test_self_contradicting_runtime_is_rejected_even_without_a_snapshot(self):
        # min > current accepts no consumer at all, so there is nothing for a snapshot to
        # decide and its absence is no reason to let the runtime through.
        with _runtime(_lib_with_window(5, 4), built_abi=None):
            with self.assertRaises(ImportError) as ctx:
                abi_check.check_librbln_abi(_FAKE_PATH)
        message = str(ctx.exception)
        self.assertIn("inconsistent ABI window", message)
        self.assertIn("no ABI snapshot recorded", message)

    def test_opt_out_skips_a_mismatch_that_would_otherwise_raise(self):
        for value in ("1", "ON", "on", "true", "yes"):
            with (
                patch.dict(os.environ, {abi_check._SKIP_ENV: value}),
                patch.object(abi_check, "get_built_abi", return_value=99),
                # The opt-out has to cover every step, taking the handle included, or it
                # cannot unblock a machine where that step is what fails.
                patch.object(abi_check, "open_mapped_runtime", side_effect=AssertionError("opened")),
            ):
                self.assertEqual(
                    abi_check.check_librbln_abi(_FAKE_PATH),
                    abi_check.VERDICT_SKIPPED_DISABLED,
                    f"value={value!r}",
                )

    def test_opt_out_ignores_empty_and_off_values(self):
        for value in ("", "0", "off", "  "):
            with patch.dict(os.environ, {abi_check._SKIP_ENV: value}):
                self.assertFalse(abi_check.is_abi_check_disabled(), f"value={value!r}")


def _mapped_librbln(case):
    """A handle on the librbln.so the import path maps, or skip if it is not installed."""
    try:
        path = rbln_runtime_lib.load_runtime_library()
    except FileNotFoundError as e:
        case.skipTest(f"librbln.so not installed: {e}")
    return abi_check.open_mapped_runtime(path)


@pytest.mark.test_set_ci
class TestInstalledCombination(TestCase):
    """The librbln.so installed on this machine against the wheel under test."""

    def test_the_mapped_library_gives_a_usable_handle(self):
        # The handshake reads symbols off this handle; a None would silently turn every
        # check into the pre-ABI fail-open path.
        lib = _mapped_librbln(self)
        self.assertIsNotNone(lib)
        self.assertTrue(getattr(lib, "_name", "").endswith("librbln.so"))

    def test_this_build_and_this_runtime_agree(self):
        built = abi_check.get_built_abi()
        if built is None:
            self.skipTest("this torch-rbln recorded no ABI snapshot (pre-handshake rebel-compiler)")
        lib = _mapped_librbln(self)
        window = abi_check.read_runtime_abi(lib)
        if window is None:
            self.skipTest("installed librbln.so predates the ABI handshake")
        self.assertIsNone(
            abi_check.abi_mismatch_reason(built, window[0], window[1]),
            f"installed librbln.so accepts {window[0]}..{window[1]} but this build recorded {built}",
        )


if __name__ == "__main__":
    run_tests()
