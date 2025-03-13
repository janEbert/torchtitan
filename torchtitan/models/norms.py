# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn


def build_norm(norm_type: str, dim: int, eps: float = 1e-6):
    """
    Builds the specified normalization layer based on the norm_type.

    Args:
        norm_type (str): The type of normalization layer to build.
            Supported types: layernorm, np_layernorm, rmsnorm
        dim (int): The dimension of the normalization layer.
        eps (float, optional): The epsilon value for numerical stability. Defaults to 1e-6.

    Returns:
        The built normalization layer.

    Raises:
        NotImplementedError: If an unknown norm_type is provided.
    """
    norm_type = norm_type.lower()  # Normalize to lowercase

    if norm_type == "layernorm":
        return nn.LayerNorm(dim, eps=eps, bias=False)
    elif norm_type == "np_layernorm":
        return nn.LayerNorm(dim, eps=eps, elementwise_affine=False, bias=False)
    elif norm_type == "rmsnorm":
        return RMSNorm(dim, eps=eps, weight=True)
    elif norm_type == "np_rmsnorm":
        return RMSNorm(dim, eps=eps, weight=False)
    else:
        raise NotImplementedError(f"Unknown norm_type: '{norm_type}'")


class RMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self, dim: int, eps: float = 1e-6, weight: bool = True):
        super().__init__()
        self.eps = eps
        if weight:
            self.weight = torch.nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x: torch.Tensor):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            return output * self.weight
        return output

    def forward(self, x: torch.Tensor):
        output = self._norm(x.float()).type_as(x)
        return output

    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)  # type: ignore
