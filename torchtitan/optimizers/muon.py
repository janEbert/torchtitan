import math

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate, Shard


@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """

    def __init__(
        self,
        full_params,
        model,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        dp_mesh=None,
        **kwargs,
    ):

        self.fsdp_enabled = dp_mesh is not None
        self.dp_mesh = dp_mesh  # DeviceMesh for DP communication

        print(
            f"Muon optimizer is enabled with dp_mesh={dp_mesh} | fsdp_enabled={self.fsdp_enabled}"
        )

        muon_params, adamw_params = [], []

        for name, p in model.named_parameters():
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name:
                muon_params.append(p)
            else:
                adamw_params.append(p)

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False

    def gather_full_grad(self, g):
        """Gathers the full gradient across all distributed processes using DTensor."""
        if not self.fsdp_enabled:
            return g  # No sharding, return the original gradient

        assert isinstance(g, DTensor), "Expected gradient to be a DTensor"

        replicated_grad = g.redistribute(
            placements=[Replicate()] * g.device_mesh.ndim
        )  # make sure all rank has the same shape
        return replicated_grad

    def shard_grad(self, u, rank, world_size):
        """Extracts the correct shard for the current rank from a replicated DTensor."""
        if not self.fsdp_enabled:
            return u

        chunks = list(torch.chunk(u, world_size, dim=0))  # Split across GPUs
        return chunks[rank]
        # return u.redistribute(placements=[Shard(0)])

    def update_momentum(self, p, g, momentum):
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(g)
        buf = state["momentum_buffer"]
        buf.mul_(momentum).add_(g)
        # if nesterov:
        #     g = g.add(buf, alpha=momentum)
        # else:
        #     g = buf
        # return g

    def get_gradient_from_state(self, p, nestroev=False, momentum=0):
        buf = self.state[p]["momentum_buffer"]
        if nestroev:
            g = p.grad.add(buf, alpha=momentum)
        else:
            g = buf
        return g

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # we first update the momentum and then apply the update
        for group in self.param_groups:
            for p in group["params"]:
                # sanity check
                g = p.grad
                if g is None or self.state[p]["use_muon"] is False:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                self.update_momentum(p, g, group["momentum"])

        # then its the actual muon update
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]
            nestroev = group["nesterov"]
            ns_steps = group["ns_steps"]

            # generate weight updates in distributed fashion
            for p in group["params"]:
                # sanity check
                if p.grad is None or self.state[p]["use_muon"] is False:
                    continue
                g = self.get_gradient_from_state(p, nestroev, momentum)
                ############################
                # Step 1: Gather full gradient across ranks
                full_g = self.gather_full_grad(g)
                if self.fsdp_enabled:
                    full_g = full_g.to_local()

                ############################
                # Step 2: Run the Newton-Schulz iteration
                u = zeropower_via_newtonschulz5(full_g, steps=ns_steps)

                ############################
                # Step 3: Extract the correct shard for this rank
                u = self.shard_grad(u, rank, world_size)

                # Convert back to DTensor with replicated placement
                if self.fsdp_enabled:
                    u = DTensor.from_local(
                        u, device_mesh=g.device_mesh, placements=g.placements
                    )

                ############################
                # Step 4: update the parameter
                with torch.no_grad():
                    # apply weight decay and update parameters
                    p.data.mul_(1 - lr * wd).add_(u, alpha=-lr)

        ############################
        #       AdamW backup       #
        ############################

        self.update_adamw()

        return loss

    def update_adamw(
        self,
    ):
        ############################
        #       AdamW backup       #
        ############################
        for group in self.param_groups:
            # Get all parameters that don't use muon and have gradients
            params = [
                p
                for p in group["params"]
                if not self.state[p]["use_muon"] and p.grad is not None
            ]

            if not params:
                continue

            lr = group["lr"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            # Process parameters one by one - simple, reliable approach
            for p in params:
                g = p.grad
                state = self.state[p]

                # Initialize state if needed
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g, device=p.device)
                    state["moment2"] = torch.zeros_like(g, device=p.device)

                state["step"] += 1
                step = state["step"]

                # Get momentum buffers
                buf1 = state["moment1"]
                buf2 = state["moment2"]

                # Update momentum buffers - safer than lerp
                buf1.mul_(beta1).add_(g, alpha=1 - beta1)
                buf2.mul_(beta2).add_(
                    g * g, alpha=1 - beta2
                )  # More reliable than g.square()

                # Compute bias corrections
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                step_size = lr / (bias_correction1 / math.sqrt(bias_correction2))

                # Apply weight decay and update parameters
                with torch.no_grad():
                    p.data.mul_(1 - lr * weight_decay)
                    p.data.addcdiv_(buf1, buf2.sqrt().add_(eps), value=-step_size)
