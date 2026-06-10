# Owner(s): ["module: PrivateUse1"]

"""Minimal repro for the REBEL cross-chiplet v2v rejection (PR #85).

In vLLM eager prefill, KV-cache strided writes failed with
``rbln_memcpy_v2v(_multi) -> rc=1`` for layers whose pool sits past the first
chiplet (~34 GiB), then the engine died SYS_EBUSY. This reproduces it without
vLLM. Required ingredients (each verified necessary):

  - src built from a compiled-op output (matmul, physical bf16) replicated via
    ``expand().contiguous()`` (a stride-0 v2v); ``randn``/``repeat`` src is fine
  - chiplet 0 filled by multiple lazy pools, each first-touched by a strided
    write; one big touched pool or untouched fillers don't trigger it
  - the failing write targets a pool past the chiplet boundary

The strided-v2v CPU fallback (this PR) gates the rejection, so the write must
land correct values either way. Bit-exactness still pins the fallback path; a
runtime fix simply makes the copies native again.
"""

import pytest
import torch

import torch_rbln  # noqa: F401

from test.utils import is_rebel_device

GIB = 1 << 30


@pytest.mark.single_worker
@pytest.mark.skipif(not is_rebel_device(), reason="cross-chiplet repro needs REBEL (>35 GiB, chiplets)")
def test_strided_write_past_chiplet_boundary():
    n_kv, partition, head = 8, 1024, 64

    mm = torch.matmul(
        torch.randn(128, 512, dtype=torch.float16, device="rbln"),
        torch.randn(512, head, dtype=torch.float16, device="rbln"),
    )
    src = mm.reshape(1, 1, 128, head).expand(n_kv, 1, 128, head).contiguous()
    src_cpu = src.to("cpu")

    def kv_pool(num_blocks):
        return torch.empty(2, num_blocks, n_kv, 1, partition, head, dtype=torch.float16, device="rbln")

    def kv_write(pool):
        pool[0, torch.tensor(1, dtype=torch.int32), :, :, 0:16, :] = src[:, :, :16, :]

    # Fill chiplet 0 with strided-touched 1 GiB pools, then write past the boundary.
    pools = []
    for _ in range(34):
        pool = kv_pool(512)
        kv_write(pool)
        pools.append(pool)

    deep = kv_pool(512)
    assert deep.data_ptr() >= 34 * GIB, "expected the deep pool past the first chiplet"
    kv_write(deep)  # rejected by the runtime today; CPU fallback must absorb it

    torch.testing.assert_close(deep[0, 1, :, :, 0:16, :].to("cpu"), src_cpu[:, :, :16, :], rtol=0, atol=0)


if __name__ == "__main__":
    raise SystemExit("Run via pytest: pytest test/rbln/test_v2v_cross_chiplet.py")
