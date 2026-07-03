# Owner(s): ["module: PrivateUse1"]

"""Host-backed dummy device contract (RBLN_DUMMY_DEVICE=1).

Dummy mode is an explicit opt-in that lets torch-rbln construct device tensors
and run memory transfers on host memory, so a model can be traced/compiled on a
host with no NPU. It is forced regardless of physical NPU presence.

RBLN_DUMMY_DEVICE must be set before ``import torch_rbln`` (the device-mapping
singleton reads it once at init), so each case runs in a fresh subprocess with
the env set.
"""

import os
import subprocess
import sys
import textwrap

import pytest


def _clean_env() -> dict:
    """A hermetic env copy for subprocess tests.

    Strips external state these tests must not inherit:
      - TORCH_RBLN_DIAGNOSE: another test module (test_find_and_load_tvm_library)
        sets it at import (it disables backend init) and never restores it, so it
        leaks into the pytest process and then into ``os.environ`` here.
      - RBLN_DEVICE_MAP / RBLN_NPUS_PER_DEVICE: a parent shell/CI mapping would
        change the dummy logical device count; tests assume the default of 1 and
        set their own map via ``env_extra`` when they need one.
      - RBLN_TARGET_SOC / RBLN_DEVICES: both feed get_npu_name() and thus the
        resolved compile target; a parent-set value would flip the no-NPU cases
        below. Tests set these explicitly via ``env_extra`` when they need them.
    """
    env = dict(os.environ)
    for key in (
        "TORCH_RBLN_DIAGNOSE",
        "RBLN_DEVICE_MAP",
        "RBLN_NPUS_PER_DEVICE",
        "RBLN_TARGET_SOC",
        "RBLN_DEVICES",
    ):
        env.pop(key, None)
    return env


def _run_with_dummy(snippet: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = _clean_env()
    env["RBLN_DUMMY_DEVICE"] = "1"
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(snippet)],
        env=env,
        capture_output=True,
        text=True,
        errors="replace",  # compiler logs may contain non-UTF-8 bytes
    )


def _assert_ok(proc: subprocess.CompletedProcess) -> None:
    assert proc.returncode == 0, f"subprocess failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"


def test_reports_logical_device_but_no_physical():
    proc = _run_with_dummy(
        """
        import torch, torch_rbln
        # default count is 1 (no RBLN_DEVICE_MAP set)
        assert torch.rbln.device_count() == 1, torch.rbln.device_count()
        assert torch.rbln.is_available() is True
        # physical count must not query the runtime in dummy mode; reports 0.
        assert torch.rbln.physical_device_count() == 0, torch.rbln.physical_device_count()
        print("OK")
        """
    )
    _assert_ok(proc)
    assert "OK" in proc.stdout


def test_construct_device_tensor_with_value():
    # The exact pattern that fails on a no-NPU host without dummy mode:
    # torch.tensor(-inf, device='rbln:0') in vLLM's logits processor __init__.
    proc = _run_with_dummy(
        """
        import math, torch, torch_rbln
        t = torch.tensor(-math.inf, device="rbln:0")
        assert t.device.type == "rbln"
        back = t.cpu().item()
        assert math.isinf(back) and back < 0, back
        # device='rbln' (no index) resolves to the current device
        t2 = torch.tensor([1.0, 2.0, 3.0], device="rbln")
        assert t2.cpu().tolist() == [1.0, 2.0, 3.0]
        print("OK")
        """
    )
    _assert_ok(proc)
    assert "OK" in proc.stdout


def test_factories_and_scalar_readback():
    proc = _run_with_dummy(
        """
        import torch, torch_rbln
        assert torch.zeros(4, device="rbln:0").cpu().tolist() == [0.0, 0.0, 0.0, 0.0]
        assert torch.full((2, 2), 7.0, device="rbln:0").cpu().tolist() == [[7.0, 7.0], [7.0, 7.0]]
        assert torch.arange(0, 5, device="rbln:0").cpu().tolist() == [0, 1, 2, 3, 4]
        assert torch.tensor(42, device="rbln:0").item() == 42
        x = torch.full((2,), 3.0, device="rbln:0")
        assert x.clone().cpu().tolist() == [3.0, 3.0]
        print("OK")
        """
    )
    _assert_ok(proc)
    assert "OK" in proc.stdout


def test_set_device_and_context_do_not_raise():
    # With device_count() >= 1 the count-guards in torch_rbln.device pass exactly
    # like a real device, so no dummy-specific Python branch is needed.
    proc = _run_with_dummy(
        """
        import torch, torch_rbln
        torch.rbln.set_device(0)
        assert torch.rbln.current_device() == 0
        with torch.rbln.device(0):
            _ = torch.tensor(1.0, device="rbln")
        print("OK")
        """
    )
    _assert_ok(proc)
    assert "OK" in proc.stdout


def test_device_count_honors_device_map_group_count():
    proc = _run_with_dummy(
        """
        import torch, torch_rbln
        # 3 logical-device groups -> 3 dummy logical devices.
        assert torch.rbln.device_count() == 3, torch.rbln.device_count()
        torch.tensor(1.0, device="rbln:2")  # highest index must be usable
        print("OK")
        """,
        env_extra={"RBLN_DEVICE_MAP": "[0],[1],[2]"},
    )
    _assert_ok(proc)
    assert "OK" in proc.stdout


def test_device_map_preserves_tp_shape():
    # RBLN_DEVICE_MAP group sizes must survive so torch.compile's auto TP sizing
    # still works (topology keeps non-empty physical-id lists in dummy mode).
    proc = _run_with_dummy(
        """
        import torch, torch_rbln
        from torch_rbln._internal.rsd_utils import (
            auto_determine_num_devices,
            get_physical_device_ids,
        )
        assert get_physical_device_ids(0) == [0, 1], get_physical_device_ids(0)
        assert auto_determine_num_devices(0) == 2
        assert auto_determine_num_devices(1) == 2
        print("OK")
        """,
        env_extra={"RBLN_DEVICE_MAP": "[0,1],[2,3]"},
    )
    _assert_ok(proc)
    assert "OK" in proc.stdout


def test_boolean_spellings():
    # RBLN_DUMMY_DEVICE is a boolean flag (shared with the rebel runtime); the
    # logical count comes from RBLN_DEVICE_MAP (default 1), not this value. Cover
    # the full truthy/falsy spellings the parser recognizes.
    for truthy in ("1", "true", "t", "on", "yes", "y"):
        proc = _run_with_dummy(
            "import torch, torch_rbln; "
            "assert torch.rbln.is_dummy_device() is True; "
            "assert torch.rbln.device_count() == 1; print('OK')",
            env_extra={"RBLN_DUMMY_DEVICE": truthy},
        )
        _assert_ok(proc)
        assert "OK" in proc.stdout

    for falsy in ("0", "false", "f", "off", "no", "n"):
        env = _clean_env()
        env["RBLN_DUMMY_DEVICE"] = falsy
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch, torch_rbln; assert torch.rbln.is_dummy_device() is False; print('OK')",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        _assert_ok(proc)
        assert "OK" in proc.stdout


@pytest.mark.parametrize("invalid", ["4", "2", "maybe"])
def test_invalid_value_fails_fast(invalid):
    # RBLN_DUMMY_DEVICE is boolean, NOT an integer device count. A non-boolean
    # value is rejected loudly by the rebel runtime at startup (process aborts on
    # import) rather than silently disabling dummy mode — pin that contract.
    env = _clean_env()
    env["RBLN_DUMMY_DEVICE"] = invalid
    proc = subprocess.run(
        [sys.executable, "-c", "import torch, torch_rbln"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "boolean" in (proc.stdout + proc.stderr).lower()


def test_is_dummy_device_api():
    proc = _run_with_dummy("import torch, torch_rbln; assert torch.rbln.is_dummy_device() is True; print('OK')")
    _assert_ok(proc)
    assert "OK" in proc.stdout


def test_is_dummy_device_false_without_env():
    env = _clean_env()
    env.pop("RBLN_DUMMY_DEVICE", None)
    proc = subprocess.run(
        [sys.executable, "-c", "import torch, torch_rbln; assert torch.rbln.is_dummy_device() is False; print('OK')"],
        env=env,
        capture_output=True,
        text=True,
    )
    _assert_ok(proc)
    assert "OK" in proc.stdout


def test_device_map_rejects_duplicate_physical_id():
    # Duplicate ids are rejectable without hardware; dummy must reject like real.
    proc = _run_with_dummy(
        "import torch, torch_rbln; torch.rbln.device_count()",
        env_extra={"RBLN_DEVICE_MAP": "[0],[0]"},
    )
    assert proc.returncode != 0
    assert "more than one logical device" in (proc.stdout + proc.stderr).lower()


def test_device_map_rejects_invalid_group_size():
    # TP=3 is not an allowed group size; dummy must reject it like real hardware
    # so a config that compiles under dummy also runs on an NPU.
    proc = _run_with_dummy(
        "import torch, torch_rbln; torch.rbln.device_count()",
        env_extra={"RBLN_DEVICE_MAP": "[0,1,2]"},
    )
    assert proc.returncode != 0
    assert "valid sizes" in (proc.stdout + proc.stderr).lower()


def test_torch_compile_tracing_smoke():
    # Dummy mode's purpose is the no-NPU compile path. A custom backend that does
    # not lower to the device confirms dynamo can fakify and trace over dummy rbln
    # tensors (full rebel compilation is execution-triggered and validated via the
    # vLLM VLLM_RBLN_COMPILE_ONLY flow, which writes artifacts without executing).
    proc = _run_with_dummy(
        """
        import torch, torch_rbln
        def backend(gm, example_inputs):
            return gm.forward  # eager replay; ops fall back to CPU in dummy mode
        f = torch.compile(lambda x: x * 2 + 1, backend=backend, fullgraph=True)
        x = torch.ones(4, device="rbln:0")
        assert f(x).cpu().tolist() == [3.0, 3.0, 3.0, 3.0]
        print("OK")
        """
    )
    _assert_ok(proc)
    assert "OK" in proc.stdout


def test_set_device_layout_like_is_noop():
    # Runtime-free in dummy mode: same-device whole allocations -> no-op (no NPU).
    proc = _run_with_dummy(
        """
        import torch, torch_rbln
        target = torch.empty(4, device="rbln:0")
        ref = torch.empty(4, device="rbln:0")
        torch.rbln.set_device_layout_like(target, ref)
        print("OK")
        """
    )
    _assert_ok(proc)
    assert "OK" in proc.stdout


def test_offload_is_noop():
    # torch.rbln.offload() must not touch the runtime in dummy mode.
    proc = _run_with_dummy(
        """
        import torch, torch_rbln
        with torch.rbln.offload():
            t = torch.tensor([1.0, 2.0], device="rbln:0")
        assert t.cpu().tolist() == [1.0, 2.0]
        print("OK")
        """
    )
    _assert_ok(proc)
    assert "OK" in proc.stdout


@pytest.mark.parametrize("npus_per_device", [1, 2, 4, 8])
def test_npus_per_device_sets_single_group_tp(npus_per_device):
    # RBLN_NPUS_PER_DEVICE=N (no RBLN_DEVICE_MAP) -> one logical device of size N (TP=N).
    # Sizes valid on both dev and deploy builds; the group is host-only indices in
    # dummy mode, so no NPU is needed to materialize a width-N device.
    proc = _run_with_dummy(
        """
        import os
        import torch, torch_rbln
        from torch_rbln._internal.rsd_utils import auto_determine_num_devices, get_physical_device_ids
        n = int(os.environ["RBLN_NPUS_PER_DEVICE"])
        assert torch.rbln.device_count() == 1, torch.rbln.device_count()
        assert get_physical_device_ids(0) == list(range(n)), get_physical_device_ids(0)
        assert auto_determine_num_devices(0) == n
        torch.tensor(1.0, device="rbln:0")  # device of size N is usable
        print("OK")
        """,
        env_extra={"RBLN_NPUS_PER_DEVICE": str(npus_per_device)},
    )
    _assert_ok(proc)
    assert "OK" in proc.stdout


@pytest.mark.parametrize(
    ("device_map", "expected_count", "expected_ids0"),
    [
        ("[0]", 1, [0]),  # single device, TP=1
        ("[0],[1]", 2, [0]),  # two devices, TP=1 each
        ("[0,1]", 1, [0, 1]),  # single device, TP=2
        ("[0,1,2,3]", 1, [0, 1, 2, 3]),  # single device, TP=4
        ("[0,1],[2,3]", 2, [0, 1]),  # two devices, TP=2 each
    ],
)
def test_device_map_shapes(device_map, expected_count, expected_ids0):
    # RBLN_DEVICE_MAP shape must survive into the dummy topology: the logical-device
    # count and per-device TP shape (physical ids of device 0) match real hardware,
    # so a config that compiles under dummy also runs on an NPU.
    proc = _run_with_dummy(
        """
        import os
        import torch, torch_rbln
        from torch_rbln._internal.rsd_utils import get_physical_device_ids
        count = int(os.environ["EXP_COUNT"])
        ids0 = [int(x) for x in os.environ["EXP_IDS0"].split(",")]
        assert torch.rbln.device_count() == count, torch.rbln.device_count()
        assert get_physical_device_ids(0) == ids0, get_physical_device_ids(0)
        torch.tensor(1.0, device=f"rbln:{count - 1}")  # highest index is usable
        print("OK")
        """,
        env_extra={
            "RBLN_DEVICE_MAP": device_map,
            "EXP_COUNT": str(expected_count),
            "EXP_IDS0": ",".join(str(i) for i in expected_ids0),
        },
    )
    _assert_ok(proc)
    assert "OK" in proc.stdout


# A physical device id past any realistic host's range. Mapping user device 0 ->
# this id makes rblnGetDeviceInfo() miss, so get_npu_name(0) returns None -- i.e.
# a *genuine no-NPU host*, reproduced even on a machine that physically has NPUs.
# RBLN_DUMMY_DEVICE keeps tensor/alloc host-backed; only the NPU-name probe is
# forced to miss. Without dummy, this same setup dies at the first device op (see
# test_dummy_is_required_no_silent_cpu), which is the intended behavior.
_NO_NPU_DEVICES = "999"


_COMPILE_ONLY_SNIPPET = """
import glob, os, tempfile
import torch, torch_rbln
from rebel._C import get_npu_name

# Precondition: this really is a no-NPU view (get_npu_name misses).
assert get_npu_name(0) is None, get_npu_name(0)

cache = tempfile.mkdtemp(prefix="rbln_dummy_compile_")
opts = {{"mode": ["compile_only"], "cache_dir": cache}}
npu = os.environ.get("RBLN_DUMMY_TEST_NPU")
if npu:
    opts["npu"] = npu


class Net(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x) * 2.0 + 1.0


m = Net().eval().to("rbln:0")
x = torch.tensor([-1.0, 0.0, 1.0, 2.0], device="rbln:0")
try:
    torch.compile(m, backend="rbln", dynamic=False, options=opts)(x)
except Exception as e:
    print("COMPILE_EXC:", type(e).__name__, str(e))
arts = glob.glob(os.path.join(cache, "*.rbln"))
{assertion}
print("OK")
"""


@pytest.mark.parametrize("npu", ["RBLN-CA25", "RBLN-CR03"])
def test_compile_only_targets_npu_option_on_no_npu_host(npu):
    # No NPU (dummy + out-of-range RBLN_DEVICES => get_npu_name is None) and no
    # RBLN_TARGET_SOC: the compile target is resolved *solely* from the
    # torch.compile `npu` option, and a .rbln artifact is written for that SoC.
    proc = _run_with_dummy(
        _COMPILE_ONLY_SNIPPET.format(assertion="assert len(arts) == 1 and os.path.getsize(arts[0]) > 0, arts"),
        env_extra={"RBLN_DUMMY_TEST_NPU": npu, "RBLN_DEVICES": _NO_NPU_DEVICES},
    )
    _assert_ok(proc)
    assert "OK" in proc.stdout
    # The artifact is compiled for the requested SoC, not some probed/default one.
    assert f"Target NPU: {npu}" in (proc.stdout + proc.stderr), proc.stderr


def test_compile_only_without_npu_option_fails_fast_on_no_npu_host():
    # No NPU, no `npu` option, no RBLN_TARGET_SOC: nothing identifies a target
    # SoC, so ensure_valid_npu raises instead of silently defaulting -- no
    # artifact is produced.
    proc = _run_with_dummy(
        _COMPILE_ONLY_SNIPPET.format(assertion="assert not arts, arts"),
        env_extra={"RBLN_DEVICES": _NO_NPU_DEVICES},
    )
    _assert_ok(proc)
    assert "OK" in proc.stdout
    # Failed for the right reason: unresolved target SoC, not some other error.
    assert "Please specify `npu`" in proc.stdout, proc.stdout


def test_dummy_is_required_no_silent_cpu():
    # The out-of-range RBLN_DEVICES leaves 0 logical devices. WITHOUT dummy mode
    # the first device op must fail loudly (no silent host fallback) -- dummy is
    # an explicit opt-in, not an implicit no-NPU shim.
    env = _clean_env()
    env.pop("RBLN_DUMMY_DEVICE", None)
    env["RBLN_DEVICES"] = _NO_NPU_DEVICES
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import torch, torch_rbln
                try:
                    torch.tensor([1.0], device="rbln:0")
                except RuntimeError as e:
                    assert "no logical device" in str(e), str(e)
                    print("OK")
                else:
                    raise AssertionError("expected RuntimeError: no logical device")
                """
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    _assert_ok(proc)
    assert "OK" in proc.stdout


# Decoder (embedding + causal self-attention with a KV cache + MLP + LM head)
# shared by the compile (dummy) and run (real device) steps below. Same seed =>
# identical weights in both subprocesses, so the NPU argmax can be checked against
# the CPU reference. Two graphs mirror real inference: prefill (whole prompt) and
# decode (one token, consuming the prefill KV cache).
_DECODER_MODEL = """
torch.manual_seed(0)
_VOCAB, _D, _H, _HD, _L = 64, 32, 4, 8, 6


class _Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(_VOCAB, _D)
        self.ln = torch.nn.LayerNorm(_D)
        self.qkv = torch.nn.Linear(_D, 3 * _D)
        self.o = torch.nn.Linear(_D, _D)
        self.ln2 = torch.nn.LayerNorm(_D)
        self.fc1 = torch.nn.Linear(_D, 4 * _D)
        self.fc2 = torch.nn.Linear(4 * _D, _D)
        self.lnf = torch.nn.LayerNorm(_D)
        self.head = torch.nn.Linear(_D, _VOCAB)

    def _qkv(self, h):
        b, t = h.shape[0], h.shape[1]
        q, k, v = self.qkv(h).split(_D, dim=2)
        return tuple(z.view(b, t, _H, _HD).transpose(1, 2) for z in (q, k, v))

    def _tail(self, x, a):
        b, t = a.shape[0], a.shape[2]
        x = x + self.o(a.transpose(1, 2).reshape(b, t, _D))
        x = x + self.fc2(torch.nn.functional.gelu(self.fc1(self.ln2(x))))
        return self.head(self.lnf(x))

    def prefill(self, ids):
        x = self.emb(ids)
        q, k, v = self._qkv(self.ln(x))
        a = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self._tail(x, a), k, v

    def decode(self, ids, past_k, past_v):
        x = self.emb(ids)
        q, k, v = self._qkv(self.ln(x))
        k = torch.cat([past_k, k], dim=2)
        v = torch.cat([past_v, v], dim=2)
        a = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        return self._tail(x, a), k, v


_IDS = [list(range(1, _L + 1))]
_DTOK = [[7]]
"""

_DECODER_COMPILE = (
    """
import glob, os
import torch, torch_rbln
from rebel._C import get_npu_name

npu = get_npu_name(0)
if npu is None:  # genuine no-NPU host: nothing to run the artifacts on -> skip
    print("SKIP_NO_NPU")
    raise SystemExit(0)
print("NPU", npu)
"""
    + _DECODER_MODEL
    + """
m = _Decoder().eval().to("rbln:0")
opt = {{"mode": ["compile_only"], "cache_dir": {cache!r}, "npu": npu}}
try:  # compile_only writes each artifact; the call itself errors (no real device)
    torch.compile(m.prefill, backend="rbln", dynamic=False, options=opt)(
        torch.tensor(_IDS, device="rbln:0"))
except Exception:
    pass
try:  # distinct past_k/past_v so the graph matches the run (aliasing would specialize)
    pk = torch.zeros(1, _H, _L, _HD, device="rbln:0")
    pv = torch.zeros(1, _H, _L, _HD, device="rbln:0")
    torch.compile(m.decode, backend="rbln", dynamic=False, options=opt)(
        torch.tensor(_DTOK, device="rbln:0"), pk, pv)
except Exception:
    pass
assert len(glob.glob(os.path.join({cache!r}, "*.rbln"))) == 2, "expected 2 artifacts (prefill+decode)"
print("OK")
"""
)

_DECODER_RUN = (
    """
import glob, os
import torch, torch_rbln
"""
    + _DECODER_MODEL
    + """
m = _Decoder().eval()
# CPU reference (same seed => same weights as the compiled model)
lp, k, v = m.prefill(torch.tensor(_IDS))
ld, _, _ = m.decode(torch.tensor(_DTOK), k, v)
want = (int(lp[0, -1].argmax()), int(ld[0, -1].argmax()))

m = m.to("rbln:0")
opt = {{"cache_dir": {cache!r}, "npu": {npu!r}}}
prefill = torch.compile(m.prefill, backend="rbln", dynamic=False, options=opt)
decode = torch.compile(m.decode, backend="rbln", dynamic=False, options=opt)
before = set(glob.glob(os.path.join({cache!r}, "*.rbln")))
lp2, k2, v2 = prefill(torch.tensor(_IDS, device="rbln:0"))
ld2, _, _ = decode(torch.tensor(_DTOK, device="rbln:0"), k2, v2)
after = set(glob.glob(os.path.join({cache!r}, "*.rbln")))
assert not (after - before), "recompiled instead of loading the dummy-built artifacts"
got = (int(lp2[0, -1].argmax().cpu()), int(ld2[0, -1].argmax().cpu()))
assert got == want, (got, want)
print("OK")
"""
)


def test_dummy_compiled_prefill_decode_runs_on_real_npu(tmp_path):
    # End-to-end purpose of dummy mode: prefill/decode graphs compiled with no NPU
    # must load and run correctly on a real NPU. Compile both graphs (compile-only)
    # in a dummy subprocess, then load + run prefill->decode (KV cache handed from
    # prefill into decode) in a non-dummy (real device) subprocess and check the
    # argmax equals the CPU reference. NPU presence is detected inside the
    # subprocesses (get_npu_name), so this module stays free of a parent-process
    # torch_rbln import; skipped with no NPU.
    cache = str(tmp_path / "cache")

    proc = _run_with_dummy(_DECODER_COMPILE.format(cache=cache))
    _assert_ok(proc)
    if "SKIP_NO_NPU" in proc.stdout:
        pytest.skip("no NPU available to compile/run the decoder against")
    assert "OK" in proc.stdout
    npu = next(ln.split(maxsplit=1)[1] for ln in proc.stdout.splitlines() if ln.startswith("NPU "))

    env = _clean_env()  # non-dummy: the real NPU loads and runs the dummy-built graphs
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_DECODER_RUN.format(cache=cache, npu=npu))],
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
    )
    _assert_ok(proc)
    assert "OK" in proc.stdout


# NOTE: overlap/self-copy safety of the dummy v2v path (memmove, not memcpy) is
# covered in test/cpp/core/RBLNDummyDeviceTest.cpp::V2VHandlesOverlap — PyTorch's
# copy_ guards against aliasing storage before dispatch, so the overlap cannot be
# driven from the Python layer.


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
