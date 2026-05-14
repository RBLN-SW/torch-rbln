import torch

from torch_rbln._internal.compile_cache import compile_rbln_cached
from torch_rbln._internal.log_utils import rbln_log_debug


class OpModule_transpose(torch.nn.Module):
    def forward(self, *args, **kwargs):
        return torch.transpose(*args, **kwargs)


_transpose_op_module = OpModule_transpose().eval()


def custom_transpose_rbln(self, dim0: int, dim1: int, out=None):
    from torch_rbln.device.context_holder import out_tensor_context

    result_tensor = out
    assert out.is_contiguous()  # assume which transpose output is always contiguous
    rbln_log_debug(f"custom_transpose_rbln: result_shape={result_tensor.shape}")

    with out_tensor_context(result_tensor):
        compiled = compile_rbln_cached(
            _transpose_op_module,
            dynamic=False,
            options={"disable_logger": True, "tensor_parallel_size": 1},
            device_cache_key=self.device.index,
        )
        external_result = compiled(self, dim0=dim0, dim1=dim1)
        if result_tensor is None:
            result_tensor = external_result
        elif isinstance(external_result, torch.Tensor) and (external_result.data_ptr() != result_tensor.data_ptr()):
            result_tensor.copy_(external_result)

    # finalize_output_tensor(out, result_tensor, result_shape, tuple(self), {})

    return result_tensor
