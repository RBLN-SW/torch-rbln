"""Run environment diagnostics for loading torch-rbln on the rbln runtime.

Usage:
  python -m torch_rbln.diagnose

  Running this module skips torch-rbln's backend initialization, so it works even when
  ``import torch_rbln`` fails. ``TORCH_RBLN_DIAGNOSE=1`` does the same for any other entry point.

Use when ``import torch_rbln`` fails with:
  ImportError: torch-rbln runs on the rbln runtime, but `import rbln.runtime` failed
or
  ImportError: RBLN ABI mismatch

This prints where the ``rbln`` package imports from, the ``librbln_rt.so`` it maps, the ABI id
this build recorded against the one the runtime reports, REBEL_HOME and the other variables that
decide which runtime is picked up, and the GCC that built each native library.
"""

import sys


def main() -> int:
    from torch_rbln._internal.env_diagnostic import print_diagnostics

    print("Running torch-rbln environment diagnostics...", file=sys.stderr)
    print_diagnostics(verbose=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
