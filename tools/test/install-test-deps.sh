#!/bin/bash
# =============================================================================
# install-test-deps.sh — Install dependencies needed to run torch-rbln tests
# =============================================================================
#
# Target users: External developers who want to run the full test suite without
# manually installing dependencies per test group.
#
# Assumptions:
#   - torch-rbln is already installed (e.g. via tools/dev-setup.sh).
#   - rebel-compiler is installed and usable.
#
# Steps (each can be skipped via env var if a CI runner already provides it):
#   1. Test runner   : pytest, pytest-xdist
#   2. Test infra    : expecttest (for torch.testing._internal)
#   3. Model tests   : torchvision (PyTorch CPU index), pandas, transformers 4
#                      (the line test_transformers.py's models load under)
#   4. inference stack: vllm-rbln (git clone + editable) and, through its
#                      dependencies, optimum-rbln, in their own venv. That
#                      stack pins transformers 5, so it cannot share the test
#                      venv; test/run_tests.py runs test_optimum_llm.py and
#                      test_vllm_llm.py with this venv's interpreter. The venv
#                      sees the test venv's site-packages (torch, torch-rbln,
#                      rebel-compiler, pytest) through a .pth file and adds
#                      only the stack on top. The packages under test must stay
#                      the test venv's copies, so the install is verified
#                      against that before the script exits.
#
# Usage:
#   ./tools/test/install-test-deps.sh [--dry-run]
#
# Optional environment:
#   UV=1                 Use ``uv pip install`` instead of ``python -m pip``.
#   VLLM_RBLN_REPO       Override the source repo (default rbln-sw/vllm-rbln).
#   VLLM_RBLN_REF        Override the ref (default origin/ci/torch-rbln-model-tests).
#   VLLM_RBLN_DIR        Override the local checkout path
#                        (default ``$PROJECT_ROOT/vllm-rbln``).
#   INFERENCE_VENV       Override the inference-stack venv path
#                        (default ``$PROJECT_ROOT/.venv-inference``, where
#                        test/run_tests.py looks for it).
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(realpath "$(dirname "$0")")"
readonly SCRIPT_DIR
PROJECT_ROOT="$(realpath "${SCRIPT_DIR}/../..")"
readonly PROJECT_ROOT

# ----- arg parsing ----------------------------------------------------------

DRY_RUN=0
for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      sed -n '2,30p' "$0"
      exit 0
      ;;
    *) echo "Unknown option: ${arg}" >&2; exit 1 ;;
  esac
done

# ----- helpers --------------------------------------------------------------

run() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] $*"
    return 0
  fi
  "$@"
}

pip_install() {
  pip_install_into python "$@"
}

# pip_install_into <python> <pip args...>: install into the venv of <python>.
pip_install_into() {
  local py="$1"
  shift
  if [[ "${UV:-0}" = "1" ]]; then
    run uv pip install --python "${py}" "$@"
  else
    run "${py}" -m pip install "$@"
  fi
}

log_step() {
  echo
  echo "=== $* ==="
}

# ----- step 1: pytest -------------------------------------------------------

install_test_runner() {
  log_step "Test runner (pytest, pytest-xdist)"
  pip_install pytest pytest-xdist
}

# ----- step 2: test infra ---------------------------------------------------

install_test_infra() {
  log_step "Test infra (expecttest)"
  pip_install "expecttest>=0.3.0,<0.4.0"
}

# ----- step 3: model-test deps (torchvision + pandas) -----------------------
#
# torchvision needs the PyTorch CPU index (the +cpu wheel must come from
# download.pytorch.org rather than PyPI). pandas is a plain PyPI package used
# only by test/models/test_optimum_llm.py.
#
# transformers stays on the 4 line here: test_transformers.py loads EXAONE-3.5's
# hub modeling code, which no transformers 5 release runs. optimum-rbln is not
# installed here; every release that supports this torch pins transformers 5, so
# it lives in the inference venv (step 4).

install_model_test_deps() {
  log_step "Model-test deps (torchvision CPU + pandas + transformers 4)"
  pip_install "torchvision==0.25.0+cpu" \
    --index-url https://download.pytorch.org/whl/cpu \
    --force-reinstall \
    --no-deps
  pip_install "pandas==2.2.3"
  pip_install "transformers<5"
}

# ----- step 4: inference stack (vllm-rbln + optimum-rbln) --------------------
#
# The stack goes into its own venv because it pins transformers 5 (see step 3).
# The venv layers on this interpreter's site-packages through a .pth file, so
# torch, torch-rbln, rebel-compiler and pytest are shared and only vllm-rbln
# plus its runtime dependencies (optimum-rbln among them) are added. Those
# dependencies are read from vllm-rbln's pyproject.toml so nothing is re-pinned
# here. torch-rbln and torch are left out and the editable install is
# ``--no-deps``: vllm-rbln bounds torch-rbln to a release line that a nightly or
# editable torch-rbln does not satisfy, and a resolving install would replace
# the package under test.

vllm_wheel_index_from_pyproject() {
  # Read the vllm-cpu index URL out of vllm-rbln's pyproject.toml so we don't
  # have to keep a separate vllm version pinned in this script.
  local pyproject="$1"
  python - "${pyproject}" <<'PY'
import sys, tomllib, pathlib
data = tomllib.loads(pathlib.Path(sys.argv[1]).read_text())
for idx in data.get("tool", {}).get("uv", {}).get("index", []):
    if idx.get("name") == "vllm-cpu":
        print(idx["url"])
        sys.exit(0)
sys.exit(f"vllm-cpu index URL not found in {sys.argv[1]}")
PY
}

vllm_rbln_runtime_deps() {
  # One requirement per line, environment markers kept.
  local pyproject="$1"
  python - "${pyproject}" <<'PY'
import re, sys, tomllib, pathlib
data = tomllib.loads(pathlib.Path(sys.argv[1]).read_text())
skip = {"torch-rbln", "torch"}
for req in data["project"]["dependencies"]:
    name = re.split(r"[\s\[<>=!~;]", req.strip(), maxsplit=1)[0].lower().replace("_", "-")
    if name not in skip:
        print(req)
PY
}

install_vllm_rbln() {
  local repo="${VLLM_RBLN_REPO:-https://github.com/rbln-sw/vllm-rbln.git}"
  local ref="${VLLM_RBLN_REF:-origin/ci/torch-rbln-model-tests}"
  local dir="${VLLM_RBLN_DIR:-${PROJECT_ROOT}/vllm-rbln}"
  local venv="${INFERENCE_VENV:-${PROJECT_ROOT}/.venv-inference}"
  local py="${venv}/bin/python"

  log_step "Inference stack: vllm-rbln (clone + editable install at ${ref}) into ${venv}"

  if [[ ! -x "${py}" ]]; then
    run python -m venv "${venv}"
  fi
  # A venv this script did not create may have no pip (``uv venv`` makes one
  # without), and the install below runs the inference venv's own pip.
  if [[ "${DRY_RUN}" -eq 0 ]] && ! "${py}" -m pip --version >/dev/null 2>&1; then
    run "${py}" -m ensurepip
  fi
  # site.addsitedir rather than a bare path: it also processes the .pth files
  # *inside* the test venv, which is how an editable install (torch-rbln built
  # from this checkout, an external rebel-compiler) puts its source tree on the
  # path. It appends, so the inference venv's own packages keep priority and
  # transformers 5 still wins there.
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] write ${venv}/<purelib>/torch-rbln-test-venv.pth -> addsitedir(this interpreter's purelib)"
  else
    python - "${py}" <<'PY'
import pathlib, subprocess, sys, sysconfig
parent = sysconfig.get_paths()["purelib"]
child = subprocess.check_output(
    [sys.argv[1], "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"], text=True
).strip()
pathlib.Path(child, "torch-rbln-test-venv.pth").write_text(
    f"import site; site.addsitedir({parent!r})\n"
)
PY
  fi

  if [[ ! -d "${dir}/.git" ]]; then
    echo "Cloning ${repo} into ${dir}..."
    run git clone "${repo}" "${dir}"
  fi
  echo "Checking out ${ref} in ${dir}..."
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] (cd ${dir} && git fetch origin --prune && git checkout --detach ${ref})"
  else
    (cd "${dir}" && git fetch origin --prune && git checkout --detach "${ref}")
  fi

  local vllm_index
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    vllm_index="<resolved-from-pyproject-at-runtime>"
  else
    vllm_index="$(vllm_wheel_index_from_pyproject "${dir}/pyproject.toml")"
    echo "Resolved vllm wheel index from vllm-rbln pyproject: ${vllm_index}"
  fi

  # The rbln index resolves vllm-rbln's ``optimum-rbln`` pin; ordinary PyPI
  # packages come from the default index.
  local -a deps
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    deps=("<vllm-rbln runtime deps minus torch-rbln/torch, from pyproject at runtime>")
  else
    local deps_text
    deps_text="$(vllm_rbln_runtime_deps "${dir}/pyproject.toml")"
    mapfile -t deps <<< "${deps_text}"
  fi
  # Installed with the inference venv's own pip, never uv, whatever UV is set
  # to: pip reads the target interpreter's sys.path, so it sees the test venv's
  # torch and torch-rbln through the .pth above and leaves them alone. uv
  # resolves against the venv directory only, does not see them, and pulls its
  # own torch in as a transitive dependency of vllm.
  run "${py}" -m pip install "${deps[@]}" \
    --extra-index-url "${vllm_index}" \
    --extra-index-url https://pypi.rbln.ai/simple/ \
    --extra-index-url https://download.pytorch.org/whl/cpu
  run "${py}" -m pip install -e "${dir}" --no-deps

  verify_shared_packages "${py}"
}

# The inference venv exists to add the vLLM stack, not to re-create the packages
# under test. A second copy of torch or torch-rbln there would be a different
# binary from the one the rest of the suite exercises, and the extension built
# against the test venv's torch would load the inference venv's. Fail here
# rather than let a run report on packages nobody meant to test.
verify_shared_packages() {
  local py="$1"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] verify torch / torch_rbln / rebel resolve to this interpreter"
    return 0
  fi

  python - "${py}" <<'PY'
import json
import os
import subprocess
import sys
import sysconfig
import tempfile

# -I so the probe measures the interpreter's own environment: no PYTHONPATH, and
# no working directory on sys.path (a torch-rbln checkout has a ``torch`` symlink
# at its root, and the installer is normally run from there).
PROBE = r"""
import json, os
out = {}
for name in ("torch", "torch_rbln", "rebel", "transformers"):
    try:
        mod = __import__(name)
    except Exception as exc:
        out[name] = [None, repr(exc)]
    else:
        path = getattr(mod, "__file__", "") or ""
        out[name] = [getattr(mod, "__version__", None), os.path.realpath(path) if path else ""]
print(json.dumps(out))
"""


def probe(interpreter):
    # Absolute: the probe runs from a directory that is not the caller's.
    interpreter = os.path.abspath(interpreter)
    with tempfile.TemporaryDirectory() as neutral:
        return json.loads(subprocess.check_output([interpreter, "-I", "-c", PROBE], text=True, cwd=neutral))


parent = os.path.realpath(sysconfig.get_paths()["purelib"])
child = probe(sys.argv[1])
here = probe(sys.executable)

problems = []
for name in ("torch", "torch_rbln", "rebel"):
    version, path = child[name]
    if version is None:
        problems.append("{}: not importable in the inference venv ({})".format(name, path))
        continue
    if path != here[name][1]:
        problems.append("{}: {} in the inference venv, {} here".format(name, path, here[name][1]))
    if here[name][0] is not None and version != here[name][0]:
        problems.append("{}: {} in the inference venv, {} here".format(name, version, here[name][0]))

# The whole point of the split: the inference venv must bring its own transformers.
if child["transformers"][0] is None:
    problems.append("transformers: not importable in the inference venv ({})".format(child["transformers"][1]))
elif child["transformers"][1].startswith(parent):
    problems.append("transformers: inference venv falls back to the test venv's copy ({})".format(child["transformers"][1]))

if problems:
    sys.exit("inference venv does not share the packages under test:\n  " + "\n  ".join(problems))

shared = ", ".join("{} {}".format(n, child[n][0]) for n in ("torch", "torch_rbln", "rebel"))
print("Shared with the inference venv: " + shared)
print("Inference venv transformers: {}".format(child["transformers"][0]))
PY
}

# ----- main -----------------------------------------------------------------

install_test_runner
install_test_infra
install_model_test_deps
install_vllm_rbln

echo
echo "All test dependencies installed."
