import functools
from typing import Optional

import torch
from torch.distributed.tensor import DTensor, Replicate
import torch.nn as nn

INIT_FN_TYPES = ["trunc_normal", "normal", "orthogonal", "scion_normal"]


# Deliberately throw away `mean` and `std` arguments.
def _wrap_ignore_mean_std(fn):

    @functools.wraps(fn)
    def wrapped_fn(tensor, mean=None, std=None, *args, **kwargs):
        return fn(tensor, *args, **kwargs)

    return wrapped_fn


def orthogonal_(parameters, gain: float = 1.0, generator: Optional[torch.Generator] = None):
    assert generator is not None, "distributed orthogonal init needs an RNG seed"
    # Force all ranks to use same RNG.
    rng_state = generator.get_state()
    torch.distributed.broadcast(rng_state, group_src=0)
    generator.set_state(rng_state)

    if isinstance(parameters, nn.Parameter):
        tensor = parameters.data
    else:
        tensor = parameters

    if not isinstance(tensor, DTensor):
        return nn.init.orthogonal_(tensor, gain=gain, generator=generator)

    buffer = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
    nn.init.orthogonal_(buffer, gain=gain, generator=generator)

    buffer = DTensor.from_local(
        buffer,
        placements=[Replicate()] * tensor.device_mesh.ndim,
        device_mesh=tensor.device_mesh,
    ).redistribute(placements=tensor.placements)

    tensor.data = buffer


def scion_normal_(
        tensor,
        mean: float = 0.0,
        std: float = 1.0,
        norm_axis: int = 1,
        eps: float = 1e-12,
):
    nn.init.normal_(
        tensor,
        mean=mean,
        std=std,
    )
    with torch.no_grad():
        divisor = torch.rsqrt(tensor.pow(2).sum(axis=norm_axis, keepdim=True) + eps)
        tensor.mul_(divisor)


def build_init_fn(init_fn_type: str):
    """
    Builds the specified initialization function based on `init_fn_type`.

    Args:
        init_fn_type (str): The type of normalization layer to build.
            Supported types: trunc_normal, normal

    Returns:
        The built initialization function.

    Raises:
        NotImplementedError: If an unknown `init_fn_type` is provided.
    """
    init_fn_type = init_fn_type.lower()  # Normalize to lowercase

    if init_fn_type == "trunc_normal":
        return nn.init.trunc_normal_
    elif init_fn_type == "normal":
        return nn.init.normal_
    elif init_fn_type == "zeros":
        return _wrap_ignore_mean_std(nn.init.zeros_)
    elif init_fn_type == "orthogonal":
        return _wrap_ignore_mean_std(orthogonal_)
    elif init_fn_type == "scion_normal":
        return scion_normal_
    else:
        raise NotImplementedError(f"Unknown `init_fn_type`: '{init_fn_type}'")
