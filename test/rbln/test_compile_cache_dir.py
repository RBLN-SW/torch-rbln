# Owner(s): ["module: PrivateUse1"]

"""Eager ops, and graphs compiled with no ``cache_dir`` option, keep their compiled functions in
the compile cache directory, from which another process takes them rather than compiling again,
and an eager op without capturing it again.
"""

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import pytest
import torch
from rebel import v2
from torch.testing._internal.common_utils import run_tests, TestCase

from torch_rbln._internal import compile_cache
from torch_rbln._internal.compile_cache import _stored_key
from torch_rbln._internal.env_utils import compile_cache_dir
from torch_rbln._internal.ops_utils import _PlainOp, get_view_op_module


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Eager ops, then a graph, each compiled for the first time in the process; prints how many
# functions the compile cache found, how many it compiled and how many eager ops were captured.
_COMPILES = f"""
    import sys
    sys.path.insert(0, {_PROJECT_ROOT!r})
    import torch
    from rebel import v2
    from torch_rbln._internal.env_utils import compile_cache_dir

    capture, captures = v2.frontend.capture, []
    def counted(*args, **kwargs):
        captures.append(args)
        return capture(*args, **kwargs)
    v2.frontend.capture = counted

    x = torch.randn(4, 64, dtype=torch.float16).to("rbln")
    y = torch.relu(x + x) * x
    torch.compile(lambda a: torch.sin(a) + a, backend="rbln")(y).cpu()
    info = v2.cache_info(compile_cache_dir())
    print("compile cache", info.hits, info.misses, len(captures))
"""

# An eager op and a graph, compiled and run on the device, whatever the compile cache directory is.
_RUNS = f"""
    import sys
    sys.path.insert(0, {_PROJECT_ROOT!r})
    import torch
    import torch_rbln

    x = torch.randn(4, 64, dtype=torch.float16)
    eager = x.to("rbln") * 3 + x.to("rbln")
    graph = torch.compile(lambda a: torch.sin(a) + a, backend="rbln")(x.to("rbln"))
    torch.testing.assert_close(eager.cpu(), x * 3 + x, atol=0.01, rtol=0.01)
    torch.testing.assert_close(graph.cpu(), torch.sin(x) + x, atol=0.01, rtol=0.01)
    print("ran", eager.device, graph.device)
"""


class _Scaled(torch.nn.Module):
    """An op whose function holds state: the tensor it makes."""

    def forward(self, x):
        return x * torch.tensor([1.0, 2.0, 3.0, 4.0] * 16, dtype=x.dtype)


@pytest.mark.test_set_ci
class TestCompileCacheDir(TestCase):
    def test_the_torch_rbln_variable_comes_first(self):
        with mock.patch.dict(os.environ, {"TORCH_RBLN_COMPILE_CACHE_DIR": "/own", "RBLN_COMPILE_CACHE_DIR": "/rbln"}):
            self.assertEqual(compile_cache_dir(), "/own")
            os.environ["TORCH_RBLN_COMPILE_CACHE_DIR"] = ""
            self.assertIsNone(compile_cache_dir())
            del os.environ["TORCH_RBLN_COMPILE_CACHE_DIR"]
            self.assertEqual(compile_cache_dir(), "/rbln")

    def test_unset_it_is_under_the_user_cache(self):
        with mock.patch.dict(os.environ, {"XDG_CACHE_HOME": "/xdg", "HOME": "/home/someone"}):
            for name in ("TORCH_RBLN_COMPILE_CACHE_DIR", "RBLN_COMPILE_CACHE_DIR"):
                os.environ.pop(name, None)
            self.assertEqual(compile_cache_dir(), "/xdg/torch_rbln/compile_cache")
            del os.environ["XDG_CACHE_HOME"]
            self.assertEqual(compile_cache_dir(), "/home/someone/.cache/torch_rbln/compile_cache")

    def _compiles(self, directory: Path) -> tuple[int, int, int]:
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(_COMPILES)],
            cwd=_PROJECT_ROOT,
            env={**os.environ, "TORCH_RBLN_COMPILE_CACHE_DIR": str(directory)},
            capture_output=True,
            text=True,
            timeout=600,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        # The runtime logs to stdout as well.
        (line,) = (line for line in result.stdout.splitlines() if line.startswith("compile cache "))
        hits, misses, captures = line.split()[-3:]
        return int(hits), int(misses), int(captures)

    def test_another_process_takes_the_functions_from_disk(self):
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        _, misses, captures = self._compiles(directory)
        self.assertGreater(misses, 0)
        self.assertGreater(captures, 0)
        self.assertTrue(any(directory.glob("*/*.rbln")))
        self.assertTrue(any(directory.glob("*.entry")))
        # The graph's function is found by the graph; the eager ops' are found uncaptured.
        self.assertEqual(self._compiles(directory), (1, 0, 0))

    @unittest.skipIf(os.geteuid() == 0, "root writes any directory")
    def test_ops_compile_and_run_when_the_directory_cannot_be_written(self):
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        directory.chmod(0o500)
        self.addCleanup(directory.chmod, 0o700)
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(_RUNS)],
            cwd=_PROJECT_ROOT,
            env={
                **os.environ,
                "TORCH_RBLN_COMPILE_CACHE_DIR": str(directory),
                "TORCH_RBLN_DISABLE_FALLBACK": "compile_error",
            },
            capture_output=True,
            text=True,
            timeout=600,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ran rbln:0 rbln:0", result.stdout)
        warned = [line for line in result.stderr.splitlines() if f"compile cache cannot use {directory}" in line]
        self.assertEqual(len(warned), 1, result.stderr)
        self.assertEqual(list(directory.iterdir()), [])


@pytest.mark.test_set_ci
class TestStoredOps(TestCase):
    """An eager op's function is kept on disk under a key made before its capture, of all the
    capture and the compile depend on."""

    def setUp(self):
        self.directory = self.enterContext(tempfile.TemporaryDirectory())
        self.x = torch.zeros(2, 64, dtype=torch.float16)
        self.op = get_view_op_module(torch.add, (None, None))

    def _key(self, op=None, args=None, kwargs=None, npu="RBLN-CA25", precision="device"):
        args = (self.x, self.x) if args is None else args
        return _stored_key(self.op if op is None else op, args, kwargs or {}, npu, precision)

    def test_a_change_of_any_part_of_the_key_misses(self):
        v2.api.store_entry(self._key(), b"entry", self.directory)
        self.assertEqual(v2.api.find_entry(self._key(), self.directory), b"entry")
        x = self.x
        changed = {
            "op": self._key(op=get_view_op_module(torch.sub, (None, None))),
            "view": self._key(op=get_view_op_module(torch.add, ((("permute", (1, 0)),), None))),
            "shape": self._key(args=(torch.zeros(4, 64, dtype=torch.float16), x)),
            "dtype": self._key(args=(x.bfloat16(), x.bfloat16())),
            "value": self._key(args=(x, 1.0)),
            "kwargs": self._key(kwargs={"alpha": 2}),
            "npu": self._key(npu="RBLN-CR03"),
            "precision": self._key(precision="exact"),
        }
        with mock.patch.object(torch, "__version__", "0.0"):
            changed["torch"] = self._key()
        with mock.patch.object(torch.version, "git_version", "0"):
            changed["torch build"] = self._key()
        with mock.patch.object(sys, "version", "0"):
            changed["python"] = self._key()
        with mock.patch.object(compile_cache, "_source_id", lambda: "0"):
            changed["torch-rbln"] = self._key()
        with mock.patch.object(torch, "get_default_dtype", lambda: torch.float64):
            changed["default dtype"] = self._key()
        with mock.patch.object(torch, "get_default_device", lambda: torch.device("meta")):
            changed["default device"] = self._key()
        for part, key in changed.items():
            with self.subTest(part):
                self.assertIsNone(v2.api.find_entry(key, self.directory))
        values = [self._key(args=(x, v)) for v in (1, 1.0, True, 2.0, -0.0, 0.0)]
        self.assertEqual(len(set(values)), len(values))

    def test_an_op_named_by_no_process_is_not_kept(self):
        class Local(torch.nn.Module):
            def forward(self, a, b):
                return a + b

        weighted, buffered, hooked, holding = (_PlainOp(torch.add) for _ in range(4))
        self.assertIsNotNone(self._key(op=weighted))
        weighted.weight = torch.nn.Parameter(torch.zeros(1))
        buffered.register_buffer("offset", torch.zeros(1))
        hooked.register_forward_hook(lambda *_: None)
        holding.scale = torch.ones(1)
        for op in (
            lambda a, b: a + b,
            Local(),
            get_view_op_module(lambda a, b: a + b, (None, None)),
            weighted,
            buffered,
            hooked,
            holding,
            _PlainOp(Local()),
        ):
            with self.subTest(op):
                self.assertIsNone(self._key(op=op))
        self.assertIsNone(self._key(args=(self.x, object())))
        self.assertIsNone(self._key(args=(self.x, torch.Generator())))

    def _compiled(self, op, args):
        """The function of ``op`` for ``args`` with the capture that made it, or None when the
        compile took it from disk."""
        captures = []
        capture = v2.frontend.capture

        def counted(*a, **k):
            captures.append(a)
            return capture(*a, **k)

        with (
            mock.patch.dict(os.environ, {"TORCH_RBLN_COMPILE_CACHE_DIR": self.directory}),
            mock.patch.object(v2.frontend, "capture", counted),
        ):
            compiled = compile_cache._compile(op, args, {}, args[0].device.index)
        return compiled, len(captures)

    def test_a_kept_op_runs_without_a_capture(self):
        # Distinct values, so that a maximum has one index.
        x = torch.randperm(192).reshape(3, 64).to(torch.float16) / 64
        ops = {
            "plain": (get_view_op_module(torch.mul, (None, None)), (x, x)),
            "returning a structseq": (get_view_op_module(torch.max, (None, None)), (x, 1)),
            "viewing": (get_view_op_module(torch.add, ((("permute", (1, 0)),), None)), (x.t().contiguous(), x)),
            "holding state": (_Scaled(), (x,)),
        }
        own = (*compile_cache._OWN_PACKAGES, _Scaled.__module__.partition(".")[0])
        self.enterContext(mock.patch.object(compile_cache, "_OWN_PACKAGES", own))
        for name, (op, args) in ops.items():
            with self.subTest(name):
                args = tuple(a.to("rbln") if isinstance(a, torch.Tensor) else a for a in args)
                captured, captures = self._compiled(op, args)
                self.assertEqual(captures, 1)
                kept, captures = self._compiled(op, args)
                self.assertEqual(captures, 0)
                tensors = kept.tensors(args, {})
                expected = op(*(a.cpu() if isinstance(a, torch.Tensor) else a for a in args))
                for compiled in (captured, kept):
                    result = compiled.run(tensors)
                    self.assertIs(type(result), type(expected))
                    torch.testing.assert_close(
                        torch.utils._pytree.tree_map(lambda t: t.cpu(), result), expected, atol=0.01, rtol=0.01
                    )


if __name__ == "__main__":
    run_tests()
