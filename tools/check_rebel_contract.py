#!/usr/bin/env python3
"""Report how the rebel-compiler being built against differs from what torch-rbln declares.

Runs the same check torch-rbln runs at import, so whoever moves the rebel pin sees the
divergence at configure time rather than as a slower eager path at runtime. Loaded by path
because torch_rbln cannot be imported before it is built.

Prints one ``BROKEN=`` or ``DRIFTED=`` line per divergence, and ``ERROR=`` when the check itself
could not run. Always exits 0: this reports, it does not gate a build.
"""

import importlib.util
import os
import sys
from typing import Any


_CONTRACT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "torch_rbln", "_internal", "rebel_contract.py"
)


def _load_contract() -> Any:
    spec = importlib.util.spec_from_file_location("_rebel_contract_for_build", _CONTRACT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the rebel contract from {_CONTRACT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(kind: str, text: str) -> str:
    """One ``KIND=text`` line. CMake matches per line, so the text cannot carry newlines."""
    return f"{kind}={' '.join(text.split())}"


def main() -> int:
    # Importing rebel imports torch, which autoloads the torch_rbln backend extension -- the one
    # this build is producing. On a first build it does not exist yet and the import would fail.
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

    try:
        divergences = _load_contract().verify()
    except Exception as e:  # a contract check must never be the reason a build stops
        print(_record("ERROR", f"{type(e).__name__}: {e}"))
        return 0

    for divergence in divergences:
        print(_record(divergence.kind, f"{divergence.name}: {divergence.detail}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
