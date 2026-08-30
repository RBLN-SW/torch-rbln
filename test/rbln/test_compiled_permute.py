import pytest
import torch

import torch_rbln  # noqa: F401

pytestmark = pytest.mark.skipif(not torch.rbln.is_available(), reason="needs an RBLN device")

DIMS = [0, 2, 1, 3]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.int32])
def test_compiled_permute_is_bit_exact(dtype):
    src = torch.randint(-1000, 1000, (3, 4, 16, 8)).to(dtype).to("rbln:0")
    for _ in range(2):  # the second call reuses the cached program and its output buffer
        out = torch.rbln.compiled_permute(src, DIMS)
        torch.rbln.synchronize()
        assert torch.equal(out.cpu(), src.cpu().permute(DIMS))


def test_compiled_permute_writes_a_caller_buffer():
    src = torch.randint(-1000, 1000, (3, 4, 16, 8)).to(torch.bfloat16).to("rbln:0")
    out = torch.empty((3, 16, 4, 8), dtype=torch.bfloat16, device="rbln:0")
    for _ in range(2):
        assert torch.rbln.compiled_permute(src, DIMS, out=out) is out
        torch.rbln.synchronize()
        assert torch.equal(out.cpu(), src.cpu().permute(DIMS))


def test_compiled_permute_rejects_a_mismatched_out():
    src = torch.zeros((3, 4, 16, 8), dtype=torch.bfloat16, device="rbln:0")
    with pytest.raises(ValueError, match="out must be"):
        torch.rbln.compiled_permute(
            src, DIMS, out=torch.empty((3, 4, 16, 8), dtype=torch.bfloat16, device="rbln:0")
        )


def test_compiled_permute_slots_are_independent():
    a = torch.randint(-1000, 1000, (2, 3, 4, 8)).to(torch.bfloat16).to("rbln:0")
    b = torch.randint(-1000, 1000, (2, 3, 4, 8)).to(torch.bfloat16).to("rbln:0")
    out_a = torch.rbln.compiled_permute(a, DIMS, slot=0)
    out_b = torch.rbln.compiled_permute(b, DIMS, slot=1)
    torch.rbln.synchronize()
    assert torch.equal(out_a.cpu(), a.cpu().permute(DIMS))
    assert torch.equal(out_b.cpu(), b.cpu().permute(DIMS))


@pytest.mark.parametrize("dims", [[1, 0, 2], [1, 0, 2, 3], [2, 1, 0, 3]])
def test_compiled_permute_handles_other_permutations(dims):
    shape = [2, 3, 4, 5][: len(dims)]
    src = torch.randint(-1000, 1000, shape).to(torch.bfloat16).to("rbln:0")
    out = torch.rbln.compiled_permute(src, dims)
    torch.rbln.synchronize()
    assert torch.equal(out.cpu(), src.cpu().permute(dims))


@pytest.mark.parametrize("dims", [[2, 0, 1, 3], [0, 3, 1, 2, 4]])
def test_a_miscompiled_permutation_raises_instead_of_returning_wrong_data(dims):
    """The compiler gets these wrong silently (the eager path is correct); we must not pass
    that on. Delete this test once the compiler handles them."""
    shape = [2, 3, 4, 5, 6][: len(dims)]
    src = torch.randint(-1000, 1000, shape).to(torch.bfloat16).to("rbln:0")
    with pytest.raises(RuntimeError, match="miscomputes"):
        torch.rbln.compiled_permute(src, dims)


def test_a_miscompiled_permutation_falls_back_in_copy_():
    """copy_ must still be right: it drops back to the strided walk."""
    src = torch.randint(-1000, 1000, (64, 8, 128, 64)).to(torch.bfloat16)
    dev = src.to("rbln:0")
    out = torch.empty((128, 64, 8, 64), dtype=torch.bfloat16, device="rbln:0")
    out.copy_(dev.permute([2, 0, 1, 3]))
    torch.rbln.synchronize()
    assert torch.equal(out.cpu(), src.permute([2, 0, 1, 3]))


@pytest.mark.parametrize(
    "dims, message",
    [([0, 1, 2], "permutation"), ([0, 1, 3, 2], "last")],
)
def test_compiled_permute_rejects_bad_dims(dims, message):
    src = torch.zeros((2, 3, 4, 8), dtype=torch.bfloat16, device="rbln:0")
    with pytest.raises(ValueError, match=message):
        torch.rbln.compiled_permute(src, dims)


def test_large_permuted_copy_matches_the_strided_path():
    """copy_ from a permuted source takes the compiled path above 16 MiB; result is unchanged."""
    src = torch.randint(-1000, 1000, (64, 8, 512, 64)).to(torch.bfloat16)
    dev = src.to("rbln:0")
    out = torch.empty((64, 512, 8, 64), dtype=torch.bfloat16, device="rbln:0")
    out.copy_(dev.permute(DIMS))
    torch.rbln.synchronize()
    assert torch.equal(out.cpu(), src.permute(DIMS))
