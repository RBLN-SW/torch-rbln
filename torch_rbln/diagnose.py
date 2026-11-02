"""Run environment diagnostics for RBLN library loading.

Usage:
  TORCH_RBLN_DIAGNOSE=1 python -m torch_rbln.diagnose

  (Set TORCH_RBLN_DIAGNOSE=1 so that torch/native libs are not loaded; then
  diagnose can run even when they are missing or broken.)

Use when you see:
  RuntimeError: Cannot find libraries: ['librbln.so', 'librbln_runtime.so']
or
  FileNotFoundError: Could not find librbln.so

This prints REBEL_HOME, LD_LIBRARY_PATH, PYTHONPATH, TVM paths, and which
search directories exist and contain the libraries. Helps identify
environment-override or path conflicts (e.g. site-packages vs REBEL_HOME).
"""

import sys

from torch_rbln._internal.env_diagnostic import print_diagnostics


def main() -> int:
    print("Running torch-rbln environment diagnostics...", file=sys.stderr)  # noqa: T201
    print_diagnostics(verbose=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
