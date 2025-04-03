# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import math
from typing import Any, Dict, Generic, List, TypeVar
import re
import copy
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate
from torch.optim import Optimizer, Adam, AdamW

from torchtitan.components.ft import FTManager, has_torchft
from torchtitan.config_manager import JobConfig

from optimizers import DistributedMuon, DistributedMuonV2, Muon, Scion
from torchtitan.optimizers.scion import zeropower_backends

__all__ = [
    "OptimizersContainer",
    "build_optimizers",
]

def l1_to_rms_norm(W):
    norm = torch.max(torch.norm(W.data.to(torch.float32), p=2, dim=0, dtype=torch.float32))
    scale = torch.sqrt(torch.tensor(W.data.shape[0], dtype=W.dtype, device=W.device))
    norm /= scale
    return norm

def rms_to_l1_norm(W):
    norm = torch.max(torch.norm(W.data.to(torch.float32), p=2, dim=1, dtype=torch.float32))
    scale = torch.sqrt(torch.tensor(W.data.shape[1], dtype=W.dtype, device=W.device))
    norm *= scale
    return norm

def _extract_param_groups(
    model: torch.nn.Module,
    optimizer_config: Dict[str, Any] = None,
):
    if optimizer_config is None:
        return model.parameters()

    param_groups_config = optimizer_config.pop('param_groups', None)
    if param_groups_config is not None:
        params = []
        param_dict = OrderedDict((n, p) for n, p in model.named_parameters() if p.requires_grad)

        for param_group_config in param_groups_config:
            str_match = param_group_config.pop('param_str_match')
            filter_fn = functools.partial(re.search, str_match)
            param_names = [n for n in param_dict.keys() if filter_fn(n)]
            group_params = {'params': [param_dict.pop(n) for n in param_names], 'param_names': param_names}
            assert len(group_params['params']) == len(group_params['param_names'])
            group_params.update(param_group_config)
            params.append(group_params)

        params.insert(0, {'params': param_dict.values(), 'param_names': list(param_dict.keys())})
        return params

    return model.parameters()

if has_torchft:
    import torchft as ft


T = TypeVar("T", bound=Optimizer)


class OptimizersContainer(Optimizer, Generic[T]):
    """A container for multiple optimizers.

    This class is used to wrap multiple optimizers into a single object that can be
    used to reduce the complexity of the training loop. This mimics the behavior of
    ``torch.optim.Optimizer``. This class currently only supports ``Adam`` and ``AdamW``.

    **Note**
    Users who want to customize the optimizer behavior can inherit from this class and
    extend the functionality as needed. The following methods must follow the same signature
    as ``torch.optim.Optimizer`` class: ``step()``, ``zero_grad()``, ``state_dict()``,
    ``load_state_dict()``.

    **Limitations**
    This class assumes that all the optimizers are the same type and have the same
    configurations. With this assumption, TorchTitan can support lr scheduler resharding
    (e.g., loading a checkpoint with a different number of GPUs and/or different
    parallelization strategy). Note that ``get_optimizer_state_dict`` already enables the
    resharding for the optimizer state but not for the lr scheduler state, hence the limitation.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        optimizer_kwargs (Dict[str, Any]): Keyword arguments for the optimizers.
        name (str): Name of the optimizers.
    """

    optimizers: List[T]
    model_parts: List[nn.Module]
    norm_functions = {
        'spectral_norm': lambda x: torch.linalg.norm(x.data.to(torch.float32), ord=2, dtype=torch.float32),
        'l1_to_rms': lambda x: l1_to_rms_norm(x),
        'rms_to_l1': lambda x: rms_to_l1_norm(x),
    }

    def __init__(
        self,
        model_parts: List[nn.Module],
        optimizer_cls: type[T],
        optimizer_kwargs: Dict[str, Any],
    ) -> None:
        all_params = []
        self.optimizers: List[T] = []
        self.model_parts = model_parts
        for model in self.model_parts:
            if issubclass(optimizer_cls, (Muon, DistributedMuon, DistributedMuonV2)):
                kwargs = optimizer_kwargs
                params = [p for p in model.parameters() if p.requires_grad]
                kwargs['model'] = model
            else:
                kwargs = copy.deepcopy(optimizer_kwargs) # to preserve popped objects across model parts
                params = list(_extract_param_groups(model, kwargs))
            self.optimizers.append(optimizer_cls(params, **kwargs))
            all_params.extend(params)
        self._validate_length(len(self.model_parts))
        self._post_init(all_params, kwargs) # use the latest kwargs after popping param groups

    def __iter__(self) -> Optimizer:
        return iter(self.optimizers)

    def __len__(self) -> int:
        return len(self.optimizers)

    def step(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
            optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(*args, **kwargs)

    def state_dict(self) -> Dict[str, Any]:
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        return {
            k: v
            for sd in map(func, self.model_parts, self.optimizers)
            for k, v in sd.items()
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        list(map(func, self.model_parts, self.optimizers))

    @staticmethod
    def compute_grad(p, optimizer=None, **kwargs):
        if isinstance(optimizer, Scion):
            g = p.grad
            if g is None or not p.requires_grad:
                return None
            
            momentum = kwargs.pop('momentum')
            nesterov = kwargs.pop('nesterov')
            if not optimizer.is_light and momentum != 1:
                state = optimizer.state[p]
                if 'momentum_buffer' not in state.keys():
                    raise ValueError(
                        "Momentum buffer not found in optimizer state. \
                            Please check if the optimizer is initialized correctly."
                    )
                buf = state['momentum_buffer']
                buf = buf.mul(1-momentum).add(g, alpha=momentum)
                g = buf if not nesterov else buf.mul(1-momentum).add(g, alpha=momentum)

            return optimizer.lmo(g, **kwargs)
        elif isinstance(optimizer, AdamW) or isinstance(optimizer, Adam):
            eps = kwargs['eps']
            weight_decay = kwargs['weight_decay']
            beta1, beta2 = kwargs['betas']
            assert weight_decay == 0.0, "Weight decay not supported for grad computation."

            param_optim_state = optimizer.state[p]
            step = param_optim_state['step'].item()
            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step
            denom = (param_optim_state['exp_avg_sq'].sqrt() / math.sqrt(bias_correction2)) + eps
            step_size = 1 / bias_correction1
            g = step_size * param_optim_state['exp_avg'].div(denom)

            assert isinstance(g, DTensor), "Expected gradient to be a DTensor"
            return g.redistribute(placements=[Replicate()] * g.device_mesh.ndim)
        else:
            raise NotImplementedError(f"Optimizer {optimizer.__class__.__name__} not added for grad computation.")
        
    def get_parameter_norms(self):
        norms = {}
        for i, _ in enumerate(self.model_parts):
            # NB: assumes correspondences between model parts and optimizers
            for group in self.optimizers[i].param_groups: 
                if isinstance(self.optimizers[i], Scion):
                    param_kwargs = {
                        'momentum': group['momentum'],
                        'nesterov': group['nesterov'],
                        'eps': group['eps'],
                        'norm_factor': group['norm_factor'],
                        'zeropower_backend': zeropower_backends[group['backend']],
                        'backend_steps': group['backend_steps']
                    }
                elif isinstance(self.optimizers[i], AdamW):
                    param_kwargs = {
                        'eps': group['eps'],
                        'betas': group['betas'],
                        'weight_decay': group['weight_decay']
                    }
                else:
                    raise NotImplementedError(f"Optimizer {self.optimizers[i].__class__.__name__} not added for norm computation.")
                
                for n, p in zip(group['param_names'], group['params']):
                    g = self.compute_grad(p, self.optimizers[i], **param_kwargs)
                    if g is not None:
                        p = p.redistribute(
                                placements=[Replicate()] * p.device_mesh.ndim
                            ).to_local() if isinstance(p, DTensor) else p
                        g = g.to_local() if isinstance(g, DTensor) else g
                        update = -group['lr']*g
                        if 'tok_embeddings' in n:
                            p, update = p.T, update.T
                        for norm_name, norm_func in self.norm_functions.items():
                            norms[f'model_part_{i}/{n}/param/{norm_name}'] = norm_func(p)
                            norms[f'model_part_{i}/{n}/update/{norm_name}'] = norm_func(update)
        return norms

    def get_lrs(self):
        lrs = {}
        assert len(self.model_parts) == 1, "Require that model_parts is the whole model"
        for i, group in enumerate(self.optimizers[0].param_groups): 
            lrs[f'lr/group_{i}'] = group['lr']
        return lrs
    
    def _validate_length(self, expected_length: int) -> None:
        assert expected_length == len(self.optimizers), (
            "Must pass one optimizer per model part or per param if "
            "using OptimizersInBackwardContainer."
        )

    def _post_init(
        self, all_params: list[nn.Parameter], optimizer_kwargs: dict[str, Any]
    ) -> None:
        # We need to call Optimizer.__init__() to initialize some necessary optimizer
        # functionality such as hooks.
        Optimizer.__init__(self, all_params, optimizer_kwargs)


class OptimizersInBackwardContainer(OptimizersContainer):
    """OptimizersContainer for executing ``optim.step()`` in backward pass.

    This class extend ``OptimizersContainer`` to support optimizer step in
    backward pass. ``step()`` and ``zero_grad()`` are no-op in this class.
    Instead, ``register_post_accumulate_grad_hook`` is used to register a hook to
    execute these methods when the gradient is accumulated.
    """

    def __init__(
        self,
        model_parts: List[nn.Module],
        optimizer_cls: type[T],
        optimizer_kwargs: Dict[str, Any],
    ) -> None:
        all_params = []
        self.model_parts = model_parts

        optim_dict = {}
        for model in self.model_parts:
            for p in model.parameters():
                if p.requires_grad:
                    optim_dict[p] = optimizer_cls([p], **optimizer_kwargs)
                all_params.append(p)

        def optim_hook(param) -> None:
            optim_dict[param].step()
            optim_dict[param].zero_grad()

        for model in self.model_parts:
            for param in model.parameters():
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(optim_hook)

        self.optimizers = list(optim_dict.values())

        self._validate_length(
            sum(
                len([param for param in model.parameters()])
                for model in self.model_parts
            )
        )
        self._post_init(all_params, optimizer_kwargs)

    def step(self) -> None:
        pass

    def zero_grad(self) -> None:
        pass


class FTOptimizersContainer(OptimizersContainer):
    def __init__(
        self,
        model_parts: List[nn.Module],
        optimizer_cls: type[T],
        optimizer_kwargs: Dict[str, Any],
        ft_manager: "ft.Manager",
    ) -> None:
        super().__init__(model_parts, optimizer_cls, optimizer_kwargs)

        # Force to initialize the optimizer state so that `optim.step()`
        # won't be called by state_dict() and load_state_dict().
        _ = {
            k: v
            for sd in map(get_optimizer_state_dict, model_parts, self.optimizers)
            for k, v in sd.items()
        }
        self.cache_state_dict: Dict[str, Any] = {}
        self._ft_optimizer = ft.Optimizer(ft_manager, self)
        self._call_from_ft: bool = False

    def init_cache_state_dict(self) -> None:
        self.cache_state_dict = super().state_dict()

    def state_dict(self) -> Dict[str, Any]:
        return self.cache_state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # We have to invalidate the `cache_state_dict` because optimizer uses
        # assign instead of copy when doing `load_state_dict()`. Without
        # invalidating the `cache_state_dict`, there will be memory leakage.
        self.cache_state_dict = {}
        super().load_state_dict(state_dict)
        self.init_cache_state_dict()

    def step(self, *args, **kwargs) -> None:
        """Calling the correct step() depending on the caller.

        TorchFT's OptimizerWrapper.step() is designed to be callled only once
        per train step per ft.Manager regardless how many optimizers are used.
        Hence we will need to appropriately dispatch the call.
        """
        if self._call_from_ft:
            super().step(*args, **kwargs)
        else:
            self._call_from_ft = True
            self._ft_optimizer.step(*args, **kwargs)
            self._call_from_ft = False

    def zero_grad(self, *args, **kwargs) -> None:
        """Calling the correct zero_grad() depending on the caller.

        Check the comment in ``step()``.
        """
        if self._call_from_ft:
            super().zero_grad(*args, **kwargs)
        else:
            self._call_from_ft = True
            self._ft_optimizer.zero_grad(*args, **kwargs)
            self._call_from_ft = False


def build_optimizers(
    model_parts: List[nn.Module],
    job_config: JobConfig,
    ft_manager: FTManager,
    world_mesh: DeviceMesh = None,
) -> OptimizersContainer:
    """Create a OptimizersContainer for the given model parts and job config.

    This function creates a ``OptimizersContainer`` for the given model parts.
    ``job_config`` should define the correct optimizer name and parameters.
    This function currently supports creating ``OptimizersContainer`` and
    ``OptimizersInBackwardContainer``.

    **Note**
    Users who want to customize the optimizer behavior can create their own
    ``OptimizersContainer`` subclass and ``build_optimizers``. Passing the
    customized ``build_optimizers`` to ``TrainSpec`` will create the customized
    ``OptimizersContainer``.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        job_config (JobConfig): Job config containing the optimizer name and parameters.
    """
    optim_in_bwd = job_config.optimizer.early_step_in_backward
    if optim_in_bwd and job_config.parallelism.pipeline_parallel_degree > 1:
        raise NotImplementedError(
            "Optimizers in backward is not supported with pipeline parallelism."
        )
    name = job_config.optimizer.name
    lr = job_config.optimizer.lr
    eps = job_config.optimizer.eps
    weight_decay = job_config.optimizer.weight_decay
    embed_str_match = job_config.optimizer.embed_str_match
    embed_lr = job_config.optimizer.embed_lr
    unembed_str_match = job_config.optimizer.unembed_str_match
    unembed_lr = job_config.optimizer.unembed_lr
    
    if name in ["Adam", "AdamW"] or "Muon" in name:
        optim_implementation = job_config.optimizer.implementation
        assert optim_implementation in ["fused", "foreach", "for-loop"]

        fused = optim_implementation == "fused"
        foreach = optim_implementation == "foreach"

        assert '-multiplier-' in job_config.model.flavor, "Model flavor label must contain a multiplier"
        width_multiplier = int(job_config.model.flavor.split('-multiplier-')[-1])
        
        optimizer_kwargs = {
            "lr": lr / width_multiplier,
            "eps": eps / width_multiplier,
            "betas": (0.9, 0.95),
            "weight_decay": weight_decay * width_multiplier, # WD is coupled with LR in torch AdamW
            "fused": fused,
            "foreach": foreach,
            'param_groups':
                [   { 
                        'param_str_match': embed_str_match,
                        'lr': embed_lr,
                    },
                    { 
                        'param_str_match': unembed_str_match,
                        'lr': unembed_lr / width_multiplier,
                    }
                ]
        }
    elif name == "Scion":
        backend_steps = job_config.optimizer.backend_steps
        momentum = job_config.optimizer.momentum
        nesterov = job_config.optimizer.nesterov
        is_light = job_config.optimizer.is_light
        is_unconstrained = job_config.optimizer.is_unconstrained
        optimizer_kwargs = {
                "is_light": is_light,
                "is_unconstrained": is_unconstrained,
                "lr": lr,
                'momentum': momentum,
                'nesterov': nesterov,
                'eps': eps,
                'norm_factor': 'spectral',
                'backend': 'newtonschulz5',
                'backend_steps': backend_steps,
                'param_groups':
                    [
                        {
                            'param_str_match': embed_str_match,
                            'lr': embed_lr,
                            'norm_factor': 'embed_sqrt',
                            'backend': 'identity'
                        },
                        { 
                            'param_str_match': unembed_str_match,
                            'lr': unembed_lr,
                            'norm_factor': 'unembed_sqrt',
                            'backend': 'identity'
                        }
                    ]
        }
    else:
        raise NotImplementedError(f"Optimizer {name} not added.")
    
    if "Muon" in name or name == "Scion":
        optimizer_kwargs["world_mesh"] = world_mesh

    optimizer_classes = {
        "Adam": Adam,
        "AdamW": AdamW,
        "Muon": Muon,
        "DistributedMuon": DistributedMuon,
        "DistributedMuonV2": DistributedMuonV2,
        "Scion": Scion,
    }
    if name not in optimizer_classes:
        raise NotImplementedError(f"Optimizer {name} not added.")
    optimizer_cls = optimizer_classes[name]

    if optim_in_bwd and ft_manager.enabled:
        raise ValueError("TorchFT is not supported with optimizers in backward.")
    elif optim_in_bwd:
        return OptimizersInBackwardContainer(
            model_parts, optimizer_cls, optimizer_kwargs
        )
    elif ft_manager.enabled:
        return FTOptimizersContainer(
            model_parts, optimizer_cls, optimizer_kwargs, ft_manager.manager
        )
    else:
        return OptimizersContainer(model_parts, optimizer_cls, optimizer_kwargs)
