"""Only the op under test runs on the device; everything else runs on the host.

Upstream builds samples and compares results on the device under test. On RBLN those
ops fall back or compile on their own, and a problem there hides what the op under
test did. So OpInfo samples are built on the host and moved over as the same views,
and assertEqual compares on the host unless exact_device=True. test_ops.py itself is
left as upstream has it.

Caveats: tensors a test makes itself (out= arguments) are still made on the device,
and a sample that puts a tensor on the CPU on purpose loses that. The only one today
is the CPU scalar in reference_inputs_elementwise_ternary, used by addcmul/addcdiv,
which we don't test yet.
"""

import functools

import pytest
import torch
import torch.utils._pytree as pytree
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import TestCase


@pytest.fixture(autouse=True)
def assert_equal_on_cpu(monkeypatch):
    upstream = TestCase.assertEqual

    def assert_equal(self, x, y, *args, **kwargs):
        if not kwargs.get("exact_device"):
            x, y = pytree.tree_map_only(torch.Tensor, lambda t: t.cpu() if t.device.type == "rbln" else t, (x, y))
        return upstream(self, x, y, *args, **kwargs)

    monkeypatch.setattr(TestCase, "assertEqual", assert_equal)


def _to_device(x, device):
    if isinstance(x, torch.Tensor):
        # Moving the whole storage keeps the strides and the storage offset.
        whole = x.detach().as_strided((x.untyped_storage().nbytes() // x.element_size(),), (1,), 0).to(device)
        if x._is_view():
            moved = whole.as_strided(x.size(), x.stride(), x.storage_offset())
        else:
            # Not a view of `whole`, as the sample was not one: tests such as
            # test_view_replay tell an op's views of its input by their _base.
            moved = torch.empty(0, dtype=x.dtype, device=device)
            moved.set_(whole.untyped_storage(), x.storage_offset(), x.size(), x.stride())
        return moved.requires_grad_(x.requires_grad)
    # A device the sample passes to the op, as for `to` or a factory function.
    if isinstance(x, (str, torch.device)) and str(x) == "cpu":
        return type(x)(device)
    return x


def _built_on_cpu(sample_inputs_func):
    @functools.wraps(sample_inputs_func)
    def wrapper(op_info, device, *args, **kwargs):
        if torch.device(device).type != "rbln":
            yield from sample_inputs_func(op_info, device, *args, **kwargs)
            return
        to_device = functools.partial(_to_device, device=device)
        for sample in sample_inputs_func(op_info, "cpu", *args, **kwargs):
            sample.input, sample.args, sample.kwargs = pytree.tree_map(
                to_device, (sample.input, sample.args, sample.kwargs)
            )
            yield sample

    return wrapper


@pytest.fixture(autouse=True)
def samples_on_cpu(monkeypatch):
    for op in op_db:
        for attr in ("sample_inputs_func", "reference_inputs_func"):
            if (func := getattr(op, attr)) is not None:
                monkeypatch.setattr(op, attr, _built_on_cpu(func))
