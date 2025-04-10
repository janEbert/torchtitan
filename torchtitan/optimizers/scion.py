import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate, Shard

__all__ = [
    'Scion',
]


def zeropower_via_svd(G, **kwargs):
    U, S, V = G.svd()
    X = U @ V.T
    return X


@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \\sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    # print('\n\n\n')
    # print(G.shape)
    # print('\n\n\n')
    assert len(G.shape) == 2, "Please make sure there is no biases or non-2D parameters"
    a, b, c = (3.4445, -4.7750,  2.0315)
    #     for a, b, c in [ # updated coefficients from @leloykun
    #     (4.0848, -6.8946, 2.9270),
    #     (3.9505, -6.3029, 2.6377),
    #     (3.7418, -5.5913, 2.3037),
    #     (2.8769, -3.1427, 1.2046),
    #     (2.8366, -3.0525, 1.2012),
    # ]:
    original_dtype = G.dtype
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (torch.linalg.norm(X) + eps)  # ensure top singular value <= 1

    for _ in range(steps):
        A = X @ X.T
        # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T

    return X.to(original_dtype)


zeropower_backends = dict(svd=zeropower_via_svd,
                          newtonschulz5=zeropower_via_newtonschulz5,
                          identity=lambda x, **kwargs: x)


class Scion(torch.optim.Optimizer):

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, eps=1e-7, norm_factor='none',
                 backend='newtonschulz5', backend_steps=5, dp_mesh=None):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        eps=eps, norm_factor=norm_factor,
                        backend=backend, backend_steps=backend_steps)
        self.fsdp_enabled = dp_mesh is not None
        self.dp_mesh = dp_mesh  # DeviceMesh for DP communication
        print(
            f"Scion optimizer is enabled with dp_mesh={dp_mesh} | fsdp_enabled={self.fsdp_enabled}"
        )
        super().__init__(params, defaults)

    def gather_full_grad(self, g):
        """Gathers the full gradient across all distributed processes using DTensor."""
        if not self.fsdp_enabled:
            return g  # No sharding, return the original gradient

        assert isinstance(g, DTensor), "Expected gradient to be a DTensor"

        replicated_grad = g.redistribute(
            placements=[Replicate()] * g.device_mesh.ndim
        )  # make sure all rank has the same shape
        return replicated_grad

    def shard_grad(self, g):
        """Extracts the correct shard for the current rank from a replicated DTensor."""
        if not self.fsdp_enabled:
            return g
        return g.redistribute(placements=[Shard(0)])

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            param_kwargs = {
                'momentum': group['momentum'],
                'nesterov': group['nesterov'],
                'eps': group['eps'],
                'norm_factor': group['norm_factor'],
                'zeropower_backend': zeropower_backends[group['backend']],
                'backend_steps': group['backend_steps']
            }
            for p in group['params']:
                g = self._compute_grad(p, **param_kwargs)
                if g is not None:
                    g = self.shard_grad(g)
                    if g.shape != p.data.shape:
                        raise RuntimeError(
                            f"Shape mismatch: g.shape={g.shape}, p.data.shape={p.data.shape}"
                        )
                    p.data.add_(g, alpha=-lr)
                    self.state[p]['step'] += 1

        return loss

    @torch.no_grad()
    def _compute_grad(
            self,
            p,
            momentum,
            nesterov,
            eps,
            norm_factor,
            zeropower_backend,
            backend_steps,
    ):
        g = p.grad
        if g is None or not p.requires_grad:
            return None

        # State initialization
        state = self.state[p]
        if 'step' not in state:
            state['step'] = torch.zeros((), dtype=torch.float, device=p.device)
        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = torch.zeros_like(g)

        # Compute updated gradient
        buf = state['momentum_buffer']
        buf = momentum * buf + g
        if nesterov:
            g = g + momentum * buf

        # Transform gradient with the backend function
        g = self.gather_full_grad(g)
        g = zeropower_backend(g, steps=backend_steps, eps=eps)

        # Normalize the gradient
        if norm_factor == 'spectral':
            # print('\n\n\n')
            # print('LINEAR, shape: ', g.shape)
            # print('\n\n\n')
            g = g * (g.size(0)/g.size(1))**0.5
        elif norm_factor.startswith('embed'):
            # print('\n\n\n')
            # print('EMBED, shape: ', g.shape)
            # print('\n\n\n')
            # NB: here assume shape [vocab_size, embed_dim]
            g = g * torch.rsqrt(g.pow(2).mean(1, keepdim=True) + eps)
            if norm_factor == 'embed_linear':
                g = g * g.size(1)
            elif norm_factor == 'embed_sqrt':
                g = g * g.size(1)**0.5
            else:
                raise ValueError(f"Unknown norm_factor: {norm_factor}")
        elif norm_factor.startswith('unembed'):
            # print('\n\n\n')
            # print('UNEMBED, shape: ', g.shape)
            # print('\n\n\n')
            g = g * torch.rsqrt(g.pow(2).mean(1, keepdim=True) + eps)
            if norm_factor == 'unembed_linear':
                g = g / g.size(1)
            elif norm_factor == 'unembed_sqrt':
                g = g / g.size(1)**0.5
            else:
                raise ValueError(f"Unknown norm_factor: {norm_factor}")
        elif norm_factor == 'none':
            pass
        else:
            raise ValueError(f"Unknown norm_factor: {norm_factor}")

        return g
