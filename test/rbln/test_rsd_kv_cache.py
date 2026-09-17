# Owner(s): ["module: PrivateUse1"]
"""A persistent KV cache under RSD (one logical device spanning several NPUs).

An inference engine allocates its KV cache once, in eager code, and then hands the same
buffers to two compiled programs -- prefill and decode -- which update them in place on
every step. With ``RBLN_NPUS_PER_DEVICE=2`` that buffer has to be sharded across two NPUs.
Nothing about the eager allocation decides the sharding: the buffer gets its device layout
the first time a compiled program binds it, from that program's input placement. Every
later program that binds the same buffer must agree with that placement, or the runtime
syncs the contents back to the host and re-allocates -- correct, but a host round trip of
the whole cache on every prefill<->decode alternation.

The cache is built the way an engine builds it, because the shape of the allocation is what
the sharding applies to: an untyped buffer, reinterpreted as the KV dtype and shaped
``[blocks, heads, 1, block_size, head_dim]``. The engine then reaches into that sharded
buffer from outside the graph -- whole reads, a partial read, a sub-block slot write,
block-granular staging batched through ``_foreach_copy_``, and an in-device block copy the
kernel afterwards reads through. Each of those is a different path into a buffer whose
elements live on two NPUs.

What this checks, from the torch-rbln side only:

* **lifecycle** -- after the cache is warm, steady decode runs with no real device->host
  sync, and host->device traffic and device-allocation growth stay far below the cache size
  (no re-layout, no host bounce);
* **engine-shaped access** -- a partial (single block) read agrees with the same region of
  the whole read, host->device staging lands in the blocks it addressed, and a block copied
  inside the device decodes to the same logits as the block it was copied from;
* **parity** -- every step's logits and the cache contents match the same sequence on one
  NPU (``RBLN_NPUS_PER_DEVICE=1``) and track an fp32 CPU reference.

Each configuration runs in its own subprocess because rebel snapshots ``RBLN_*`` at import.
A dummy-device case compiles both RSD programs for an ATOM target with no NPU, so the
sharded compile of this KV-cache graph is covered on any host; the real-device cases need
two ATOM NPUs (RSD is an ATOM feature; REBEL exposes one device).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest
import rebel  # noqa: F401  -- defines the rbln_custom_ops schemas the decoder calls
import torch

from test.utils import requires_physical_devices, SUPPORTED_DTYPES

from torch_rbln._internal.device_arch_utils import is_rebel_device


# --------------------------------------------------------------------------------------
# Model + cache, shared by every subprocess (same seed => same weights everywhere).
# --------------------------------------------------------------------------------------

# The cache is sized well above what a decode step legitimately allocates (its output logits
# and kernel scratch, a few tens of KiB), so that a re-allocation of the cache stands out.
VOCAB, KV_HEADS, Q_PER_KV, HD, MAXLEN = 64, 2, 4, 64, 1024
D = KV_HEADS * Q_PER_KV * HD
# One block holds the sequence; the others are what an engine stages into and copies between.
NBLOCKS, SEQ_BLOCK, COPY_BLOCK, STAGE_BLOCKS = 4, 0, 1, (2, 3)
PROMPT = list(range(1, 17))  # prefill length 16
DECODE_TOKENS = [7, 3, 11, 5]  # then 4 decode steps
# Cache positions carried back for the cross-run comparison. The sequence occupies the first
# few; the rest of the block is zero and is checked by its own count, not element by element.
REPORTED_POSITIONS = 32


def new_caches(device, dtype):
    """Allocate the KV caches the way an engine does: untyped buffers, viewed as the KV dtype.

    An engine sizes its KV pool in bytes and allocates it untyped, then reinterprets it as the
    cache dtype and shapes it ``[blocks, heads, 1, block_size, head_dim]``. That untyped buffer
    is the allocation the sharding applies to, and every access in this test -- the graph's and
    the engine's -- goes through a view of it.

    K and V get a buffer each. An engine keeps them in one allocation and hands the kernel the
    combined ``[2, blocks, ...]`` tensor; the paged-attention op here takes them as two
    arguments, and two mutated graph inputs that alias one storage are rejected by the runtime.
    """

    def one():
        elements = NBLOCKS * KV_HEADS * MAXLEN * HD
        raw = torch.zeros(elements * torch.empty((), dtype=dtype).element_size(), dtype=torch.int8, device=device)
        return raw, raw.view(dtype).view(NBLOCKS, KV_HEADS, 1, MAXLEN, HD)

    k_raw, k_cache = one()
    v_raw, v_cache = one()
    return (k_raw, v_raw), k_cache, v_cache


def step_inputs(ids, ctx, device, dtype, block=SEQ_BLOCK):
    """Program inputs for one step: token ids, context length, mask, block table."""
    t = len(ids)
    mask = torch.zeros(1, 1, 1, t, MAXLEN, dtype=dtype)
    if t > 1:  # prefill: causal over the prompt, nothing else in the cache yet
        mask[0, 0, 0] = torch.ones(t, MAXLEN, dtype=dtype).tril()
    else:  # decode: everything up to and including the new position
        mask[..., : ctx + 1] = 1.0
    table = torch.full((1,) if t > 1 else (1, 1), block, dtype=torch.int16)
    return tuple(a.to(device) for a in (torch.tensor([ids]), torch.full((1, 1), ctx, dtype=torch.int32), mask, table))


class Decoder(torch.nn.Module):
    """Embedding -> paged attention over a persistent KV cache -> MLP -> LM head.

    The attention is ``rbln_custom_ops.paged_attn_{prefill,decode}``: the kernel an engine's
    graph mode runs, which writes the new k/v into the block ``block_table`` names and attends
    over it. The caches are *inputs* of the program; ``seq`` is the context length already in
    the block, so prefill writes positions ``0..T-1`` and a decode step writes position ``seq``.
    Both phases are static-shape programs (T = prompt length, T = 1).
    """

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(VOCAB, D)
        self.ln = torch.nn.LayerNorm(D)
        self.q = torch.nn.Linear(D, D, bias=False)
        self.kv = torch.nn.Linear(D, 2 * KV_HEADS * HD, bias=False)
        self.o = torch.nn.Linear(D, D, bias=False)
        self.ln2 = torch.nn.LayerNorm(D)
        self.fc1 = torch.nn.Linear(D, 4 * D)
        self.fc2 = torch.nn.Linear(4 * D, D)
        self.lnf = torch.nn.LayerNorm(D)
        self.head = torch.nn.Linear(D, VOCAB)

    def attention(self, q, k, v, mask, k_cache, v_cache, seq, block_table):
        scale = torch.tensor(1.0 / HD**0.5)  # the kernel takes scale as a constant tensor
        prefill = q.shape[3] > 1
        op = torch.ops.rbln_custom_ops.paged_attn_prefill if prefill else torch.ops.rbln_custom_ops.paged_attn_decode
        return op(q, k, v, mask, k_cache, v_cache, seq, scale, block_table, MAXLEN)

    def forward(self, ids, seq, mask, block_table, k_cache, v_cache):
        t = ids.shape[1]
        x = self.emb(ids)
        h = self.ln(x)
        q = self.q(h).view(1, t, KV_HEADS, Q_PER_KV, HD).permute(0, 2, 3, 1, 4)  # [1, KV, Q, T, HD]
        k, v = self.kv(h).view(1, t, 2, KV_HEADS, 1, HD).permute(2, 0, 3, 4, 1, 5)  # [1, KV, 1, T, HD]
        a = self.attention(q, k, v, mask, k_cache, v_cache, seq, block_table)
        x = x + self.o(a.permute(0, 3, 1, 2, 4).reshape(1, t, D))
        x = x + self.fc2(torch.nn.functional.gelu(self.fc1(self.ln2(x))))
        return self.head(self.lnf(x))[:, -1]  # next-token logits [1, VOCAB]


class CPUDecoder(Decoder):
    """The same model with the paged-attention kernel spelled out in plain torch (fp32 reference)."""

    def attention(self, q, k, v, mask, k_cache, v_cache, seq, block_table):
        block = int(block_table.reshape(-1)[0])
        t, pos = k.shape[3], int(seq[0, 0])
        k_cache[block : block + 1, :, :, pos : pos + t] = k
        v_cache[block : block + 1, :, :, pos : pos + t] = v
        kc, vc = k_cache[block : block + 1], v_cache[block : block + 1]
        scores = torch.matmul(q, kc.transpose(3, 4)) / HD**0.5  # [1, KV, Q, T, MAXLEN]
        scores = scores.masked_fill(mask == 0, float("-inf"))
        return torch.matmul(torch.softmax(scores, dim=-1), vc)


def _cpu_reference() -> dict:
    """The same sequence in fp32 on the CPU, uncompiled."""
    m = CPUDecoder().eval()
    _, k_cache, v_cache = new_caches("cpu", torch.float32)
    with torch.no_grad():
        prefill = m(*step_inputs(PROMPT, 0, "cpu", torch.float32), k_cache, v_cache)
        decode = []
        for i, tok in enumerate(DECODE_TOKENS):
            decode.append(m(*step_inputs([tok], len(PROMPT) + i, "cpu", torch.float32), k_cache, v_cache)[0].tolist())
    return {"prefill": prefill[0].tolist(), "decode": decode}


# Runs in a subprocess: RBLN_VISIBLE_DEVICES / RBLN_NPUS_PER_DEVICE are already in its environment.
_DEVICE_DRIVER = """
import json, sys
import torch, torch_rbln
from torch_rbln._internal.rsd_utils import auto_determine_num_devices
from torch_rbln import profiler as rprof
from test.rbln.test_rsd_kv_cache import (
    COPY_BLOCK, DECODE_TOKENS, Decoder, HD, KV_HEADS, PROMPT, REPORTED_POSITIONS, SEQ_BLOCK,
    STAGE_BLOCKS, new_caches, step_inputs,
)

dtype = getattr(torch, sys.argv[1])
npus = int(sys.argv[2])
dev = torch.device("rbln:0")
assert torch.rbln.device_count() == 1, torch.rbln.device_count()
assert auto_determine_num_devices(0) == npus, (auto_determine_num_devices(0), npus)

m = Decoder().eval().to(device=dev, dtype=dtype)
run = torch.compile(m, backend="rbln", dynamic=False)
out = {}

def logits(t):
    return t.float().cpu()[0].tolist()

def used(block):
    # The positions the sequence occupies, plus how much of the rest of the block is non-zero:
    # a misplaced write shows up as either a changed value here or a changed count there.
    host = block.float().cpu()
    return {"head": host[:, :, :REPORTED_POSITIONS].flatten().tolist(), "nonzero": int((host != 0).sum())}

with torch.no_grad():
    # (A) the cache is an eager allocation; nothing has placed it on the device yet.
    raws, k_cache, v_cache = new_caches(dev, dtype)
    cache_bytes = sum(r.numel() for r in raws)

    # (B) prefill binds the cache to the first program: the cache takes that program's
    # (sharded) placement here.
    out["prefill"] = logits(run(*step_inputs(PROMPT, 0, dev, dtype), k_cache, v_cache))
    torch.rbln.synchronize()

    # (C) decode: a second program over the same cache. Warm it once (its own compile
    # and first bind), then measure a steady run of every remaining step.
    steps = [step_inputs([tok], len(PROMPT) + i, dev, dtype) for i, tok in enumerate(DECODE_TOKENS)]
    decode = [run(*steps[0], k_cache, v_cache)]
    torch.rbln.synchronize()
    alloc0 = torch.rbln.memory_stats(dev).get("allocated.total_allocated")
    with rprof.explain() as region:
        for step in steps[1:]:
            decode.append(run(*step, k_cache, v_cache))
        torch.rbln.synchronize()
    alloc1 = torch.rbln.memory_stats(dev).get("allocated.total_allocated")
    rr = region.dump()["runtime_residency"]
    out["steady"] = {
        "available": rr["available"],
        "d2h_count": rr.get("real_host_sync_d2h", {}).get("count"),
        "h2d_bytes": rr.get("real_host_sync_h2d", {}).get("bytes"),
        "cache_bytes": cache_bytes,
        "alloc_delta": None if alloc0 is None or alloc1 is None else alloc1 - alloc0,
    }
    out["decode"] = [logits(t) for t in decode]

    # (D) the accesses an engine makes to this cache from outside the graph.
    #
    # Whole read, and the same block read on its own: a partial read of a sharded buffer has
    # to gather the shards at an offset, which reading all of it does not exercise.
    out["k_block"] = used(k_cache[SEQ_BLOCK])
    whole = k_cache.float().cpu()
    out["partial_read_matches_whole"] = bool(
        torch.equal(k_cache[SEQ_BLOCK].float().cpu(), whole[SEQ_BLOCK])
    )

    # Block-granular staging, the shape a KV-transfer connector uses: per-block views of the
    # cache copied in one _foreach_copy_, host to device and back. Once the cache is bound to a
    # program it holds the device's 16-bit compute format, whose mantissa is shorter than the
    # host dtype's, so the value comes back rounded rather than byte for byte; what has to hold
    # is that each block landed in the block it was addressed to, which the error scale says.
    torch.manual_seed(2)
    staged = [torch.randn(KV_HEADS, 1, k_cache.shape[3], HD, dtype=dtype) for _ in STAGE_BLOCKS]
    torch._foreach_copy_([k_cache[b] for b in STAGE_BLOCKS], staged)
    out["staged_error"] = max(
        float((k_cache[b].float().cpu() - src.float()).abs().max() / src.float().abs().max())
        for b, src in zip(STAGE_BLOCKS, staged)
    )
    out["staging_left_sequence_block"] = used(k_cache[SEQ_BLOCK])

    # A block copied inside the device, then decoded through: the kernel must see the same
    # bytes in the copy as in the original, which comparing the two decodes proves end to end.
    torch._foreach_copy_(
        [k_cache[COPY_BLOCK], v_cache[COPY_BLOCK]], [k_cache[SEQ_BLOCK], v_cache[SEQ_BLOCK]]
    )
    out["block_copy_exact"] = bool(torch.equal(k_cache[COPY_BLOCK].cpu(), k_cache[SEQ_BLOCK].cpu()))
    pos = len(PROMPT) + len(DECODE_TOKENS)
    out["decode_from_original"] = logits(run(*step_inputs([DECODE_TOKENS[0]], pos, dev, dtype), k_cache, v_cache))
    step = step_inputs([DECODE_TOKENS[0]], pos, dev, dtype, block=COPY_BLOCK)
    out["decode_from_copy"] = logits(run(*step, k_cache, v_cache))

    # A sub-block write from eager code -- one slot of one block -- and a decode on top of it.
    slot = len(PROMPT) - 1
    torch.manual_seed(1)
    k_cache[SEQ_BLOCK, :, :, slot].copy_(torch.randn(KV_HEADS, 1, HD).to(dtype=dtype, device=dev))
    step = step_inputs([DECODE_TOKENS[0]], pos + 1, dev, dtype)
    out["after_write"] = logits(run(*step, k_cache, v_cache))
    out["k_block_after_write"] = used(k_cache[SEQ_BLOCK])

print("RESULT " + json.dumps(out))
"""

# Dummy device (no NPU): compile prefill and decode for an ATOM target with a two-NPU
# logical device, i.e. the sharded programs this test runs on real hardware.
_DUMMY_COMPILE_DRIVER = """
import glob, os, sys
import torch, torch_rbln
from torch_rbln._internal.rsd_utils import auto_determine_num_devices
from test.rbln.test_rsd_kv_cache import Decoder, new_caches, step_inputs, PROMPT, DECODE_TOKENS

dtype = getattr(torch, sys.argv[1])
cache_dir = sys.argv[2]
assert torch.rbln.is_dummy_device()
assert auto_determine_num_devices(0) == 2, auto_determine_num_devices(0)

m = Decoder().eval().to(device="rbln:0", dtype=dtype)
opt = {"mode": ["compile_only"], "cache_dir": cache_dir}
run = torch.compile(m, backend="rbln", dynamic=False, options=opt)
_, k_cache, v_cache = new_caches("rbln:0", dtype)
with torch.no_grad():
    for step in (step_inputs(PROMPT, 0, "rbln:0", dtype), step_inputs([DECODE_TOKENS[0]], len(PROMPT), "rbln:0", dtype)):
        try:  # compile_only writes the artifact; the call itself errors (no real device)
            run(*step, k_cache, v_cache)
        except Exception:
            pass
arts = glob.glob(os.path.join(cache_dir, "**", "*.rbln"), recursive=True)
assert len(arts) == 2, f"expected prefill + decode artifacts, found {arts}"
print("OK")
"""


# --------------------------------------------------------------------------------------
# Subprocess plumbing
# --------------------------------------------------------------------------------------


def _clean_env() -> dict:
    env = dict(os.environ)
    for key in (
        "RBLN_DEVICE_MAP",
        "RBLN_NPUS_PER_DEVICE",
        # Both spellings of the device selection: the runtime resolves RBLN_DEVICES over its
        # RBLN_VISIBLE_DEVICES alias, so leaving either behind would override what we pass.
        "RBLN_DEVICES",
        "RBLN_VISIBLE_DEVICES",
        "RBLN_DUMMY_DEVICE",
        "RBLN_FORCE_NPU_NAME",
    ):
        env.pop(key, None)
    # As the autouse conftest fixture does for in-process tests: a compile failure must
    # surface, not be papered over by the CPU fallback (which would pass the parity checks).
    categories = {c.strip() for c in env.get("TORCH_RBLN_DISABLE_FALLBACK", "").split(",") if c.strip()}
    env["TORCH_RBLN_DISABLE_FALLBACK"] = ",".join(sorted(categories | {"compile_error"}))
    return env


def _visible_physical_ids() -> list[str]:
    """Physical NPU ids this test may use: the selection the parent was given, else 0,1,...

    The runtime takes that selection from ``RBLN_VISIBLE_DEVICES`` or its ``RBLN_DEVICES``
    spelling, preferring the latter when both are set, so read them in the same order. Missing
    the one the parent was actually given would fall through to the ``physical_device_count()``
    range below -- but that count is already narrowed to the visible NPUs, so its ids would name
    system NPUs 0,1,... rather than the ones this process owns.
    """
    for name in ("RBLN_DEVICES", "RBLN_VISIBLE_DEVICES"):
        raw = os.environ.get(name, "")
        if raw.strip():
            return [s.strip() for s in raw.split(",") if s.strip()]
    return [str(i) for i in range(torch.rbln.physical_device_count())]


def _run(driver: str, args: list[str], env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(driver), *args],
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
    )


def _run_on_device(dtype: torch.dtype, npus: int) -> dict:
    ids = _visible_physical_ids()
    assert len(ids) >= npus, ids
    env = _clean_env()
    env["RBLN_VISIBLE_DEVICES"] = ",".join(ids[:npus])
    env["RBLN_NPUS_PER_DEVICE"] = str(npus)
    proc = _run(_DEVICE_DRIVER, [str(dtype).removeprefix("torch."), str(npus)], env)
    assert proc.returncode == 0, f"npus={npus} driver failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
    assert line is not None, proc.stdout
    return json.loads(line[len("RESULT ") :])


def _skip_unless_two_atom_npus() -> None:
    if is_rebel_device():
        pytest.skip("RSD spans NPUs; the REBEL lineup exposes one device")


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------


@pytest.mark.test_set_ci
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES, ids=lambda d: str(d).removeprefix("torch."))
def test_rsd_programs_compile_on_dummy_device(dtype, tmp_path):
    """Both KV-cache programs compile for a two-NPU ATOM logical device with no NPU present."""
    env = _clean_env()
    env.update(RBLN_DUMMY_DEVICE="1", RBLN_FORCE_NPU_NAME="RBLN-CA25", RBLN_DEVICE_MAP="[0,1]")
    proc = _run(_DUMMY_COMPILE_DRIVER, [str(dtype).removeprefix("torch."), str(tmp_path / "cache")], env)
    assert proc.returncode == 0, f"dummy compile failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "OK" in proc.stdout, proc.stdout


@pytest.mark.test_set_ci
@pytest.mark.single_worker
@requires_physical_devices(2)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES, ids=lambda d: str(d).removeprefix("torch."))
def test_sharded_kv_cache_lifecycle_and_parity(dtype):
    _skip_unless_two_atom_npus()
    rsd = _run_on_device(dtype, npus=2)
    single = _run_on_device(dtype, npus=1)
    cpu = _cpu_reference()

    # Lifecycle: steady decode over the warm, sharded cache never re-lays it out. A re-layout
    # syncs the cache to the host (a real d2h) and allocates a cache-sized replacement; the
    # legitimate per-step traffic is the step's ids/mask (a few KiB h2d) and its output logits
    # plus kernel scratch (tens of KiB), both far below a cache of hundreds of KiB.
    steady = rsd["steady"]
    assert steady["available"], "runtime residency counters unavailable; torch-rbln and librbln out of sync?"
    assert steady["d2h_count"] == 0, f"device->host sync during steady decode (cache bounced through host): {steady}"
    assert steady["h2d_bytes"] is None or steady["h2d_bytes"] < steady["cache_bytes"] // 4, steady
    assert steady["alloc_delta"] is None or steady["alloc_delta"] < steady["cache_bytes"] // 4, (
        f"device allocation grew by a cache-sized amount during steady decode (re-allocation): {steady}"
    )

    # Engine-shaped access to the sharded buffer. These are exact on any topology, so each run
    # answers for itself rather than being compared across the two.
    for name, res in (("RSD=2", rsd), ("1 NPU", single)):
        assert res["partial_read_matches_whole"], f"{name}: reading one block disagrees with reading all of it"
        # Rounding into the device's 16-bit format is a few thousandths of the data's scale; a
        # block that landed anywhere else holds unrelated values and misses by the scale itself.
        assert res["staged_error"] < 0.02, (
            f"{name}: a host->device staged block does not hold what was staged into it "
            f"(error {res['staged_error']:.3f} of the data's scale)"
        )
        assert res["block_copy_exact"], f"{name}: in-device block copy did not land byte for byte"
        assert res["staging_left_sequence_block"] == res["k_block"], (
            f"{name}: staging into spare blocks disturbed the sequence's block"
        )
        assert res["decode_from_copy"] == res["decode_from_original"], (
            f"{name}: decoding through the copied block disagrees with the block it was copied from"
        )

    # Tolerances are fractions of the reference's logit scale. Sharding changes only the
    # reduction order across NPUs, so RSD=2 and 1 NPU differ by 16-bit rounding of the same
    # programs; against the fp32 CPU reference each device run additionally carries the
    # device's 16-bit formats (ATOM's custom float is coarser than fp16) through the D-wide
    # projections and the attention softmax. A wrong or stale cache read moves logits by the
    # scale itself (the argmax flips), an order of magnitude above either bound.
    ref = torch.tensor(cpu["prefill"])
    scale = float(ref.abs().max())
    pair_tol, cpu_tol = 0.05 * scale, 0.1 * scale

    def close(a, b, tol, msg):
        torch.testing.assert_close(torch.tensor(a), torch.tensor(b), atol=tol, rtol=0.0, msg=msg)

    # Parity with one NPU: same programs, same cache protocol, the sharding must not show.
    close(rsd["prefill"], single["prefill"], pair_tol, "prefill logits: RSD=2 vs 1 NPU")
    for i, (a, b) in enumerate(zip(rsd["decode"], single["decode"])):
        close(a, b, pair_tol, f"decode step {i} logits: RSD=2 vs 1 NPU")
    close(rsd["decode_from_original"], single["decode_from_original"], pair_tol, "decode after the block copy")
    close(rsd["after_write"], single["after_write"], pair_tol, "decode after the sub-block slot write")
    for key in ("k_block", "k_block_after_write"):
        assert rsd[key]["nonzero"] == single[key]["nonzero"], f"{key}: cache occupancy differs from the 1-NPU run"
        close(rsd[key]["head"], single[key]["head"], pair_tol, f"{key}: cache contents differ from the 1-NPU run")

    # Both device runs track the fp32 CPU reference.
    for name, res in (("RSD=2", rsd), ("1 NPU", single)):
        close(res["prefill"], cpu["prefill"], cpu_tol, f"{name} prefill vs CPU")
        for i, step in enumerate(cpu["decode"]):
            close(res["decode"][i], step, cpu_tol, f"{name} decode {i} vs CPU")


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
