# Owner(s): ["module: PrivateUse1"]

"""Tests for ``torch_rbln._internal.compile_cache``.

Covers the cache key composition, hit/miss behavior, and clearing.
"""

import threading

import pytest
import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln  # noqa: F401  -- registers backend
from torch_rbln._internal import compile_cache as cc


class _PassThrough(nn.Module):
    def forward(self, *args, **kwargs):
        return args[0] if args else None


@pytest.mark.test_set_ci
class TestFreezeCacheValue(TestCase):
    """``_freeze_cache_value`` must produce hashable, deterministic keys."""

    def test_primitive_passthrough(self):
        for v in (1, 1.5, True, False, None, "abc"):
            self.assertEqual(cc._freeze_cache_value(v), v)

    def test_mapping_is_sorted_and_recursive(self):
        a = {"b": 2, "a": 1, "nested": {"y": 20, "x": 10}}
        b = {"nested": {"x": 10, "y": 20}, "a": 1, "b": 2}
        self.assertEqual(cc._freeze_cache_value(a), cc._freeze_cache_value(b))

    def test_sequence_preserves_order(self):
        self.assertNotEqual(
            cc._freeze_cache_value([1, 2]),
            cc._freeze_cache_value([2, 1]),
        )

    def test_set_is_sorted(self):
        self.assertEqual(
            cc._freeze_cache_value({3, 1, 2}),
            cc._freeze_cache_value({2, 3, 1}),
        )

    def test_callable_uses_id(self):
        f = lambda: None  # noqa: E731
        g = lambda: None  # noqa: E731
        self.assertEqual(cc._freeze_cache_value(f), ("callable", id(f)))
        self.assertNotEqual(
            cc._freeze_cache_value(f), cc._freeze_cache_value(g),
        )


@pytest.mark.test_set_ci
class TestCompileRblnCached(TestCase):
    """End-to-end cache behavior. Uses a stub model to avoid touching the device."""

    def setUp(self):
        cc.clear_rbln_compile_cache()

    def tearDown(self):
        cc.clear_rbln_compile_cache()

    def _patch_torch_compile(self, monkey_patch_call):
        """Replace ``torch.compile`` with a stub recording call args."""
        original = torch.compile
        calls = []

        def fake_compile(model, **kwargs):
            calls.append((id(model), kwargs))
            return monkey_patch_call(model, **kwargs)

        torch.compile = fake_compile
        return original, calls

    def test_cache_hit_returns_same_object(self):
        sentinel = object()
        original, calls = self._patch_torch_compile(lambda m, **kw: sentinel)
        try:
            mod = _PassThrough()
            r1 = cc.compile_rbln_cached(mod, dynamic=False, options={"a": 1})
            r2 = cc.compile_rbln_cached(mod, dynamic=False, options={"a": 1})
            self.assertIs(r1, r2)
            self.assertIs(r1, sentinel)
            self.assertEqual(len(calls), 1)
        finally:
            torch.compile = original

    def test_options_change_creates_new_entry(self):
        original, calls = self._patch_torch_compile(lambda m, **kw: object())
        try:
            mod = _PassThrough()
            r1 = cc.compile_rbln_cached(mod, dynamic=False, options={"a": 1})
            r2 = cc.compile_rbln_cached(mod, dynamic=False, options={"a": 2})
            self.assertIsNot(r1, r2)
            self.assertEqual(len(calls), 2)
        finally:
            torch.compile = original

    def test_dynamic_flag_creates_new_entry(self):
        original, calls = self._patch_torch_compile(lambda m, **kw: object())
        try:
            mod = _PassThrough()
            cc.compile_rbln_cached(mod, dynamic=False, options={})
            cc.compile_rbln_cached(mod, dynamic=True, options={})
            self.assertEqual(len(calls), 2)
        finally:
            torch.compile = original

    def test_device_cache_key_creates_new_entry(self):
        original, calls = self._patch_torch_compile(lambda m, **kw: object())
        try:
            mod = _PassThrough()
            cc.compile_rbln_cached(mod, dynamic=False, options={}, device_cache_key=0)
            cc.compile_rbln_cached(mod, dynamic=False, options={}, device_cache_key=1)
            self.assertEqual(len(calls), 2)
        finally:
            torch.compile = original

    def test_different_models_create_new_entry(self):
        original, calls = self._patch_torch_compile(lambda m, **kw: object())
        try:
            # Hold references so CPython doesn't recycle id() between calls.
            mod_a, mod_b = _PassThrough(), _PassThrough()
            self.assertNotEqual(id(mod_a), id(mod_b))
            cc.compile_rbln_cached(mod_a, dynamic=False, options={})
            cc.compile_rbln_cached(mod_b, dynamic=False, options={})
            self.assertEqual(len(calls), 2)
        finally:
            torch.compile = original

    def test_clear_cache_invalidates(self):
        original, calls = self._patch_torch_compile(lambda m, **kw: object())
        try:
            mod = _PassThrough()
            cc.compile_rbln_cached(mod, dynamic=False, options={})
            cc.clear_rbln_compile_cache()
            cc.compile_rbln_cached(mod, dynamic=False, options={})
            self.assertEqual(len(calls), 2)
        finally:
            torch.compile = original

    def test_kwargs_passed_to_torch_compile(self):
        captured = {}

        def fake_compile(model, **kwargs):
            captured.update(kwargs)
            return object()

        original = torch.compile
        torch.compile = fake_compile
        try:
            cc.compile_rbln_cached(
                _PassThrough(),
                dynamic=True,
                options={"x": 5},
            )
            self.assertEqual(captured.get("backend"), "rbln")
            self.assertIs(captured.get("dynamic"), True)
            self.assertEqual(captured.get("options"), {"x": 5})
        finally:
            torch.compile = original


if __name__ == "__main__":
    run_tests()
