# Owner(s): ["module: PrivateUse1"]

"""Eager ops, and graphs compiled with no ``cache_dir`` option, keep their compiled functions in
the compile cache directory, from which another process takes them rather than compiling again.
"""

import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from unittest import mock

import pytest
from torch.testing._internal.common_utils import run_tests, TestCase

from torch_rbln._internal.env_utils import compile_cache_dir


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Eager ops, then a graph, each compiled for the first time in the process; prints how many
# functions the compile cache found and how many it compiled.
_COMPILES = f"""
    import sys
    sys.path.insert(0, {_PROJECT_ROOT!r})
    import torch
    from rebel import v2
    from torch_rbln._internal.env_utils import compile_cache_dir

    x = torch.randn(4, 64, dtype=torch.float16).to("rbln")
    y = torch.relu(x + x) * x
    torch.compile(lambda a: torch.sin(a) + a, backend="rbln")(y).cpu()
    info = v2.cache_info(compile_cache_dir())
    print("compile cache", info.hits, info.misses)
"""


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

    def _compiles(self, directory: Path) -> tuple[int, int]:
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
        hits, misses = line.split()[-2:]
        return int(hits), int(misses)

    def test_another_process_takes_the_functions_from_disk(self):
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        hits, misses = self._compiles(directory)
        self.assertGreater(misses, 0)
        self.assertTrue(any(directory.glob("*/*.rbln")))
        self.assertEqual(self._compiles(directory), (hits + misses, 0))


if __name__ == "__main__":
    run_tests()
