import torch

__all__ = [
    'Muon',
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
    X = G.bfloat16()
    X /= (X.norm() + eps)  # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T

    return X


zeropower_backends = dict(svd=zeropower_via_svd,
                          newtonschulz5=zeropower_via_newtonschulz5,
                          identity=lambda x, **kwargs: x)


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz
    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.
    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).
    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, eps=1e-7, norm_factor='none',
                 backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        eps=eps, norm_factor=norm_factor,
                        backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

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
                    p.data.add_(g, alpha=-lr)
                    self.state[p]['step'] += 1

        return loss

    @torch.no_grad()
    def _compute_grad(self, p, momentum, nesterov, eps, norm_factor, zeropower_backend, backend_steps):
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
        buf.mul_(momentum).add_(g)
        if nesterov:
            g = g.add(buf, alpha=momentum)
        g = zeropower_backend(g, steps=backend_steps, eps=eps)
        if norm_factor == 'spectral':
            # print('\n\n\n')
            # print('LINEAR, shape: ', g.shape)
            # print('\n\n\n')
            g *= (g.size(0)/g.size(1))**0.5
        elif norm_factor.startswith('embed'):
            # print('\n\n\n')
            # print('EMBED, shape: ', g.shape)
            # print('\n\n\n')
            ### NB: here assume shape [vocab_size, embed_dim]
            g *= torch.rsqrt(g.pow(2).mean(1, keepdim=True) + eps)
            if norm_factor == 'embed_linear':
                g *= g.size(1)
            elif norm_factor == 'embed_sqrt':
                g *= g.size(1)**0.5
            else:
                raise ValueError(f"Unknown norm_factor: {norm_factor}")
        elif norm_factor.startswith('unembed'):
            # print('\n\n\n')
            # print('UNEMBED, shape: ', g.shape)
            # print('\n\n\n')
            g *= torch.rsqrt(g.pow(2).mean(1, keepdim=True) + eps)
            if norm_factor == 'unembed_linear':
                g /= g.size(1)
            elif norm_factor == 'unembed_sqrt':
                g /= g.size(1)**0.5
            else:
                raise ValueError(f"Unknown norm_factor: {norm_factor}")
        elif norm_factor == 'none':
            pass
        else:
            raise ValueError(f"Unknown norm_factor: {norm_factor}")

        return g
