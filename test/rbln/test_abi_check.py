# Owner(s): ["module: PrivateUse1"]

"""
Test the rebel ABI handshake performed when torch_rbln loads librbln.so.

Covers the accept/reject window (min_supported <= built <= current), the two
fail-open cases (pre-handshake runtime, no build-time snapshot), the opt-out
env var, and that the librbln.so installed here actually agrees with this build.
"""

import os


# Gate torch_backends_entry_point during our import of torch_rbln (it would otherwise load
# librbln.so). Restore the flag right after the import so it does not leak into other test
# modules collected on the same xdist worker (env_utils reads TORCH_RBLN_DIAGNOSE live).
_prev_diagnose = os.environ.get("TORCH_RBLN_DIAGNOSE")
os.environ["TORCH_RBLN_DIAGNOSE"] = "1"
try:
    from unittest.mock import patch

    import pytest
    from torch.testing._internal.common_device_type import instantiate_device_type_tests
    from torch.testing._internal.common_utils import run_tests, TestCase

    import torch_rbln
    from torch_rbln._internal import abi_check
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

    def test_half_exported_runtime_is_treated_as_pre_abi(self):
        # A .so with only one of the pair is malformed, but there is still no
        # window to validate against, so it must not become an import failure.
        self.assertIsNone(abi_check.read_runtime_abi(_FakeLib({"rbln_abi_current": 2})))
        self.assertIsNone(abi_check.read_runtime_abi(_FakeLib({"rbln_abi_min_supported": 2})))

    def test_no_handle_is_pre_abi(self):
        self.assertIsNone(abi_check.read_runtime_abi(None))


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
        with patch.dict(os.environ, _ABI_CHECK_ENABLED), patch.object(abi_check, "get_built_abi", return_value=2):
            self.assertEqual(abi_check.check_librbln_abi(_lib_with_window(1, 3)), abi_check.VERDICT_OK)

    def test_out_of_window_raises_with_an_actionable_report(self):
        with patch.dict(os.environ, _ABI_CHECK_ENABLED), patch.object(abi_check, "get_built_abi", return_value=5):
            with self.assertRaises(ImportError) as ctx:
                abi_check.check_librbln_abi(_lib_with_window(1, 3))
        message = str(ctx.exception)
        self.assertIn("RBLN ABI mismatch", message)
        self.assertIn("/fake/tvm/librbln.so", message)
        self.assertIn("built against rebel ABI 5", message)
        self.assertIn("python -m torch_rbln.diagnose", message)

    def test_pre_abi_runtime_warns_but_does_not_block(self):
        with patch.dict(os.environ, _ABI_CHECK_ENABLED), patch.object(abi_check, "get_built_abi", return_value=2):
            with pytest.warns(UserWarning, match="no ABI version symbols"):
                verdict = abi_check.check_librbln_abi(_FakeLib({}))
        self.assertEqual(verdict, abi_check.VERDICT_SKIPPED_PRE_ABI_RUNTIME)

    def test_missing_snapshot_warns_but_does_not_block(self):
        with patch.dict(os.environ, _ABI_CHECK_ENABLED), patch.object(abi_check, "get_built_abi", return_value=None):
            with pytest.warns(UserWarning, match="recorded no rebel ABI number"):
                verdict = abi_check.check_librbln_abi(_lib_with_window(1, 1))
        self.assertEqual(verdict, abi_check.VERDICT_SKIPPED_NO_SNAPSHOT)

    def test_opt_out_skips_a_mismatch_that_would_otherwise_raise(self):
        for value in ("1", "ON", "on", "true", "yes"):
            with (
                patch.dict(os.environ, {abi_check._SKIP_ENV: value}),
                patch.object(abi_check, "get_built_abi", return_value=99),
            ):
                self.assertEqual(
                    abi_check.check_librbln_abi(_lib_with_window(1, 1)),
                    abi_check.VERDICT_SKIPPED_DISABLED,
                    f"value={value!r}",
                )

    def test_opt_out_ignores_empty_and_off_values(self):
        for value in ("", "0", "off", "  "):
            with patch.dict(os.environ, {abi_check._SKIP_ENV: value}):
                self.assertFalse(abi_check.is_abi_check_disabled(), f"value={value!r}")


@pytest.mark.test_set_ci
class TestInstalledCombination(TestCase):
    """The librbln.so installed on this machine against the wheel under test."""

    def test_find_and_load_returns_a_usable_handle(self):
        # The handshake reads symbols off this return value; a None would silently
        # turn every check into the pre-ABI fail-open path.
        try:
            lib = torch_rbln.find_and_load_tvm_library("librbln.so")
        except FileNotFoundError as e:
            self.skipTest(f"librbln.so not installed: {e}")
        self.assertIsNotNone(lib)
        self.assertTrue(getattr(lib, "_name", "").endswith("librbln.so"))

    def test_this_build_and_this_runtime_agree(self):
        built = abi_check.get_built_abi()
        if built is None:
            self.skipTest("this torch-rbln recorded no ABI snapshot (pre-handshake rebel-compiler)")
        try:
            lib = torch_rbln.find_and_load_tvm_library("librbln.so")
        except FileNotFoundError as e:
            self.skipTest(f"librbln.so not installed: {e}")
        window = abi_check.read_runtime_abi(lib)
        if window is None:
            self.skipTest("installed librbln.so predates the ABI handshake")
        self.assertIsNone(
            abi_check.abi_mismatch_reason(built, window[0], window[1]),
            f"installed librbln.so accepts {window[0]}..{window[1]} but this build recorded {built}",
        )


instantiate_device_type_tests(TestAbiMismatchReason, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestReadRuntimeAbi, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestGetBuiltAbi, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestCheckLibrblnAbi, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestInstalledCombination, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
