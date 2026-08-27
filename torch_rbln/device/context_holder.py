from contextlib import contextmanager

from torch_rbln._internal import rebel_contract


helper = rebel_contract.eager_execution_helper()


@contextmanager
def out_tensor_context(out_tensor=None):
    """Bind the eager helper out-tensor for a compile call and always clear it."""
    if out_tensor is not None:
        helper.set_out_tensor(out_tensor)
    try:
        yield
    finally:
        helper.clear_out_tensor()
