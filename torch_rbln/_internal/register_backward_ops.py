from typing import List, Optional, Tuple  # noqa: UP035

import torch


def linear_backward_rbln(
    input: torch.Tensor, grad_output: torch.Tensor, weight: torch.Tensor, output_mask: List[bool]
) -> Tuple[Optional[torch.Tensor], ...]:
    """
    Backward pass for RBLN linear function.

    Args:
        input: Input tensor to the linear layer
        grad_output: Gradient of the output with respect to the loss
        weight: Weight matrix of the linear layer
        output_mask: Boolean mask indicating which gradients to compute
                    [input_grad, weight_grad, bias_grad]

    Returns:
        tuple: (grad_input, grad_weight, grad_bias) - gradients with respect to
                input, weight, and bias respectively. None values indicate gradients
                that were not computed based on output_mask.
    """
    # Constants for output_mask indices to improve readability
    INPUT_GRAD_IDX = 0  # Index for input gradient computation flag
    WEIGHT_GRAD_IDX = 1  # Index for weight gradient computation flag
    BIAS_GRAD_IDX = 2  # Index for bias gradient computation flag

    # Validate output_mask size
    if len(output_mask) < 3:
        raise ValueError(f"output_mask must have at least 3 elements, got {len(output_mask)}")

    grad_input = None
    grad_weight = None
    grad_bias = None
    if output_mask[INPUT_GRAD_IDX]:
        grad_input = torch.matmul(grad_output, weight)

    if output_mask[WEIGHT_GRAD_IDX]:
        # Reshape grad_output and input to 2D matrices
        # for a single matrix multiplication
        grad_output_reshaped = grad_output.reshape(-1, grad_output.shape[-1])
        input_reshaped = input.reshape(-1, input.shape[-1])

        # Perform matrix multiplication to get grad_weight
        grad_weight = torch.matmul(grad_output_reshaped.T, input_reshaped)

    if output_mask[BIAS_GRAD_IDX]:
        if grad_output.ndim == 1:
            grad_bias = grad_output
        else:
            grad_bias = grad_output.sum(dim=tuple(range(grad_output.ndim - 1)))

    return grad_input, grad_weight, grad_bias


def softmax_backward_rbln(
    grad_output: torch.Tensor, output: torch.Tensor, dim: int, input_dtype: torch.dtype
) -> torch.Tensor:
    """
    Backward pass for RBLN softmax function.

    Implements the softmax backward pass using the efficient formula:
    grad_input = output * (grad_output - sum(output * grad_output))

    Args:
        grad_output: Gradient of the output with respect to the loss
        output: Output tensor from the forward softmax operation
        dim: Dimension along which softmax was applied
        input_dtype: Data type of the input (unused but required for signature)

    Returns:
        torch.Tensor: Gradient with respect to the input
    """
    S = output * grad_output
    K = S.sum(dim=dim, keepdim=True)
    grad_input = S - output * K

    return grad_input


def silu_backward_rbln(grad_output: torch.Tensor, self: torch.Tensor) -> torch.Tensor:
    """
    Backward pass for RBLN SiLU function.

    The gradient of SiLU(x) = x * sigmoid(x) is:
    d/dx[SiLU(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))

    Args:
        grad_output: Gradient of the output with respect to the loss
        self: Input tensor to the SiLU function

    Returns:
        torch.Tensor: Gradient with respect to the input
    """
    sigmoid = torch.sigmoid(self)
    return grad_output * sigmoid * (1 + self * (1 - sigmoid))
