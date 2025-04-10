from dataclasses import dataclass
import math
from typing import NotRequired, Optional, Union, Callable

from einops import rearrange
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor, Replicate

from torchtitan.models.inits import build_init_fn
from torchtitan.models.llama.model import Attention, precompute_freqs_cis, TransformerInputsDict
from torchtitan.models.norms import build_norm
from torchtitan.protocols.train_spec import BaseModelArgs, ModelProtocol

from torchtitan.components.tokenizer import Tokenizer
from torchtitan.config_manager import JobConfig
from torchtitan.tools.logging import logger


@dataclass
class MoEModelArgs(BaseModelArgs):
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000

    max_seq_len: int = 2048
    # If `True`, then each transformer block init uses its layer ID, and if
    # `False`, each uses the total number of transformer blocks
    depth_init: bool = True
    first_in_init_fn_type: str = "normal"
    first_in_init_std: float = 1.0
    # Exponent applied to the first input layer's input dimensionality
    # to obtain its init std factor.
    first_in_exp: float = 0.0
    intermediate_init_fn_type: str = "trunc_normal"
    intermediate_init_std: float = 0.02
    # Exponent applied to the model's hidden dimensionality to obtain
    # intermediate layers' init std factors.
    intermediate_exp: float = 0.0
    # Whether to initialize the GLU gate as if it was a residual layer.
    init_gate_as_residual: bool = True
    final_out_init_fn_type: str = "trunc_normal"
    final_out_init_std: float = 1.0
    # Exponent applied to the final output layer's input dimensionality
    # to obtain its init std factor.
    final_out_exp: float = -0.5
    norm_type: str = "rmsnorm"
    qk_norm: bool = False

    use_flex_attn: bool = False

    # ==== MoE specific args ====
    n_shared_experts: int = 1
    n_routed_experts: int = 0
    activate_experts: int = 0
    moe_gate_bias_update_speed: float = 0.01
    moe_aux_loss_alpha: float = 0.01

    def update_from_config(self, job_config: JobConfig, tokenizer: Tokenizer) -> None:
        self.norm_type = job_config.model.norm_type
        self.vocab_size = tokenizer.n_words
        if job_config.model.vocab_size_multiple_of:
            vocab_divisor = job_config.model.vocab_size_multiple_of
            self.vocab_size = int(
                math.ceil(self.vocab_size / vocab_divisor) * vocab_divisor
            )
            logger.info(
                f"Padded vocab size from {tokenizer.n_words} to {self.vocab_size}."
            )
        self.max_seq_len = job_config.training.seq_len
        self.use_flex_attn = job_config.model.use_flex_attn

    def get_num_flop_per_token(self, num_params: int, seq_len: int) -> int:
        l, h, q, t = (
            self.n_layers,
            self.n_heads,
            self.dim // self.n_heads,
            seq_len,
        )
        # Reasoning behind the factor of 12 for the self-attention part of the formula:
        # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
        # 2. the flash attention does 1 more matmul recomputation in the backward
        #    but recomputation should not be counted in calculating MFU           (+0)
        # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
        # 4. we follow the convention and do not account for sparsity in causal attention
        flop_per_token = 6 * num_params + 12 * l * h * q * t

        return flop_per_token


class MoEInputsDict(TransformerInputsDict):
    tokens_list: Union[list[Optional[torch.Tensor]], torch.Tensor]
    aux_loss: NotRequired[torch.Tensor]
    moe_entropy_per_layer: NotRequired[torch.Tensor]


MoEInputs = Union[torch.Tensor, MoEInputsDict]


class Gate(nn.Module):
    def __init__(
        self, hidden_size: int, experts=8, topk=2, bias=True, bias_update_speed=0.01
    ):
        super().__init__()

        self.topk = topk
        self.experts = experts
        self.expert_embeddings = nn.Parameter(torch.randn(experts, hidden_size))
        # Expert embeddings
        self.bias = nn.Parameter(torch.zeros(experts)) if bias else None
        # Bias for load balancing
        self.bias_update_speed = bias_update_speed
        # Step size for updating bias dynamically

        # Cache for accumulated adjustments
        self._accumulated_adjustment = None
        self._num_accumulated = 0

    def __repr__(self):
        return f"Gate(experts={self.experts}, topk={self.topk}, bias={self.bias is not None})"

    def forward(self, x: torch.Tensor, update_bias: bool = False) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B*S, D)
            update_bias: Whether to accumulate bias adjustments (only during training)
        Returns:
            weights: Normalized routing weights
            indices: Selected expert indices
            scores: Expert selection scores
        """
        # Compute expert selection scores (sigmoid-based gating)
        scores = x @ self.expert_embeddings.T
        scores = torch.sigmoid(scores.to(torch.float32)).to(x.dtype)  # (B*S, N_r)

        if self.bias is not None:
            scores = scores + self.bias  # Add bias term to improve balance

        # Select top-k highest scoring experts
        top_values, indices = torch.topk(scores, self.topk, dim=-1)

        # Extract and normalize weights
        weights = scores.gather(-1, indices)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        # if update_bias and self.bias is not None:
        #     self._accumulate_adjustment(indices)

        return weights.type_as(x), indices, scores

    def _accumulate_adjustment(self, indices: torch.Tensor):
        """
        Accumulate bias adjustments with proper cross-rank synchronization.
        """
        with torch.no_grad():
            # Count local expert usage
            local_expert_counts = torch.bincount(
                indices.flatten(), minlength=self.experts
            ).float()

            # All-reduce to get global counts across all ranks
            global_expert_counts = local_expert_counts.clone()
            torch.distributed.all_reduce(
                global_expert_counts, op=torch.distributed.ReduceOp.SUM
            )

            # Normalize the expert usage based on global counts
            avg_usage = global_expert_counts.mean()  # Target average usage
            adjustment = (global_expert_counts - avg_usage) * self.bias_update_speed

            # Accumulate the adjustment
            if self._accumulated_adjustment is None:
                self._accumulated_adjustment = adjustment
            else:
                self._accumulated_adjustment += adjustment
            self._num_accumulated += 1

    def update(self):
        """
        Apply accumulated bias adjustments and clear the cache.
        """
        if self._accumulated_adjustment is not None and self._num_accumulated > 0:
            with torch.no_grad():
                if isinstance(self.bias.data, DTensor):
                    original_placements = self.bias.placements
                    original_mesh = self.bias.data.device_mesh

                    # Move bias to replicated format, then to local for adjustment
                    local_bias = self.bias.data.redistribute(
                        placements=[Replicate()]
                    ).to_local()
                    local_bias.sub_(self._accumulated_adjustment.detach())

                    # Rewrap and redistribute to original placements
                    adjusted_bias = DTensor.from_local(
                        local_bias, device_mesh=original_mesh, placements=[Replicate()]
                    )
                    self.bias.data = adjusted_bias.redistribute(
                        placements=original_placements
                    )

                else:
                    self.bias.data.sub_(self._accumulated_adjustment.detach())

            self.clear_cache()

    def clear_cache(self):
        """
        Clear the accumulated adjustments without applying them.
        """
        self._accumulated_adjustment = None
        self._num_accumulated = 0


class GroupedExperts(nn.Module):
    """This class implements the grouped experts layer used in Mixture of Experts. Each expert
    is a variant of the Gated Linear Units network. See more details in https://arxiv.org/pdf/2002.05202.

    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension.
        num_experts (int): Number of experts in this grouped experts layer. Default is 1.
        swiglu (bool): Whether to use gated linear unit. Default is True.
        activation (nn.Module): Activation function to use. Default is F.silu.
    """

    ep_mesh = None
    ep_size = None
    local_rank = None

    def __init__(
        self,
        *,
        dim_in: int,
        dim_out: int,
        num_experts: int = 1,
        swiglu: bool = True,
        activation: Callable = F.silu,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.num_experts = num_experts
        self.dim_out = dim_out

        if num_experts == 1:
            self.gate_proj = nn.Linear(dim_in, dim_out)
            self.down_proj = nn.Linear(dim_out, dim_in)
        else:
            self.gate_proj = nn.Parameter(torch.empty(num_experts, dim_in, dim_out))
            self.down_proj = nn.Parameter(torch.empty(num_experts, dim_out, dim_in))
        self.swiglu = swiglu
        if swiglu:
            if num_experts == 1:
                self.up_proj = nn.Linear(dim_in, dim_out)
            else:
                self.up_proj = nn.Parameter(torch.empty(num_experts, dim_in, dim_out))
            self.act_fn = F.silu
        else:
            self.up_proj = None
            self.act_fn = activation

    def __repr__(self):
        return f"GroupedExperts(dim_in={self.dim_in}, dim_hidden={self.dim_out}, num_experts={self.num_experts}, swiglu={self.swiglu})"

    def setup_ep(self, ep_mesh=None, ep_size=None):
        self.ep_mesh = ep_mesh
        self.ep_size = ep_size
        if self.ep_mesh is not None:
            self.local_rank = self.ep_mesh.get_local_rank()

    @staticmethod
    def dispatch_tokens(
        x, tokens_per_expert, dim_in, expert_per_rank, ep_size, ep_mesh
    ):
        """
        Dispatch tokens to corresponding experts using all_to_all.

        Args:
            x: Tensor of shape (num_experts * tokens_per_expert, dim_in)
            tokens_per_expert: int, number of tokens per expert
            dim_in: int, input dimension
            k: int, number of experts per EP rank
            ep_size: int, number of EP ranks (world size)
            ep_mesh: torch.distributed process group for EP parallel

        Returns:
            output: Tensor of shape (num_experts * tokens_per_expert, dim_in)
        """

        # x shape: (num_experts, tokens_per_expert, dim_in)
        # num_experts = expert_per_rank * ep_size
        x = x.view(
            ep_size, expert_per_rank * tokens_per_expert, dim_in
        )  # shape: (ep_size, expert_per_rank * tokens_per_expert, dim_in)

        # Create input list (1 tensor per rank, each must be contiguous)
        input_tensor_list = [x[i].contiguous() for i in range(ep_size)]

        # Prepare empty output tensors
        output_tensor_list = [
            torch.empty_like(input_tensor_list[0]) for _ in range(ep_size)
        ]

        # All-to-all communication
        torch.distributed.all_to_all(
            output_tensor_list,
            input_tensor_list,
            group=ep_mesh.get_group(),
        )

        # Concatenate output
        output = torch.cat(output_tensor_list, dim=0).view(
            expert_per_rank, tokens_per_expert * ep_size, dim_in
        )

        return output

    def gather_tokens(x, tokens_per_expert, dim_in, expert_per_rank, ep_size, ep_mesh):
        """
        Gather tokens from experts back to original shape using all_to_all.

        Args:
            x: Tensor of shape (expert_per_rank, tokens_per_expert * ep_size, dim_in)
            tokens_per_expert: int
            dim_in: int
            expert_per_rank: int
            ep_size: int
            ep_mesh: DeviceMesh

        Returns:
            output: Tensor of shape (num_experts, tokens_per_expert, dim_in)
        """
        # Reshape to (ep_size, expert_per_rank * tokens_per_expert, dim_in)
        x = x.view(ep_size, expert_per_rank * tokens_per_expert, dim_in)

        # Split into chunks to send back to each rank
        input_tensor_list = [x[i].contiguous() for i in range(ep_size)]

        # Prepare empty outputs
        output_tensor_list = [
            torch.empty_like(input_tensor_list[0]) for _ in range(ep_size)
        ]

        # Reverse all-to-all (same as forward in PyTorch)
        torch.distributed.all_to_all(
            output_tensor_list,
            input_tensor_list,
            group=ep_mesh.get_group(),
        )

        # Final reshape
        output = torch.cat(output_tensor_list, dim=0).view(
            expert_per_rank * ep_size, tokens_per_expert, dim_in
        )

        return output

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): with shape (num_experts, tokens_per_expert, dim_in) for Expert Choice(EC).

        Returns:
            torch.Tensor: with shape (num_experts, tokens_per_expert, dim_in) for Expert Choice(EC).
        """
        # Expert Choice(EC) forward
        # x shape (num_experts, tokens_per_expert, dim_in)
        num_experts, tokens_per_expert, dim_in = x.shape
        if self.ep_size is not None and self.ep_size > 1:
            # This code is used for EP
            expert_per_rank = num_experts // self.ep_size
            x = self.dispatch_tokens(
                x,
                tokens_per_expert,
                dim_in,
                expert_per_rank,
                self.ep_size,
                self.ep_mesh,
            )

        if self.num_experts > 1:
            if self.ep_size is not None and self.ep_size > 1:
                # This code is used for EP
                lee = self.local_rank * expert_per_rank
                les = (self.local_rank + 1) * expert_per_rank

                h = self.act_fn(torch.bmm(x, self.gate_proj[lee:les]))
                if self.up_proj is not None:
                    h = h * torch.bmm(x, self.up_proj[lee:les])
                out = torch.bmm(h, self.down_proj[lee:les])
                out = out.view(
                    num_experts, tokens_per_expert, dim_in
                )  # this actaull does not matter

            else:
                h = self.act_fn(torch.bmm(x, self.gate_proj))
                if self.up_proj is not None:
                    h = h * torch.bmm(x, self.up_proj)
                out = torch.bmm(h, self.down_proj)

        else:
            out = self.act_fn(self.gate_proj(x))
            if self.up_proj is not None:
                out = out * self.up_proj(x)
            out = self.down_proj(out)

        return out

    def init_weights(
            self,
            init_std: float,
            residual_div: float,
            init_gate_as_residual: bool,
            init_fn_type: str,
    ):
        init_fn = build_init_fn(init_fn_type)
        gate_init_std = (
            init_std / residual_div
            if init_gate_as_residual
            else init_std
        )
        if self.num_experts > 1:
            init_fn(self.gate_proj, mean=0.0, std=init_std / residual_div)
            init_fn(self.down_proj, mean=0.0, std=init_std / residual_div)
            if self.up_proj is not None:
                init_fn(self.up_proj, mean=0.0, std=gate_init_std)
        else:
            init_fn(self.gate_proj.weight, mean=0.0, std=init_std / residual_div)
            init_fn(self.down_proj.weight, mean=0.0, std=init_std / residual_div)
            if self.up_proj is not None:
                init_fn(self.up_proj.weight, mean=0.0, std=gate_init_std)


class FeedForward(nn.Module):
    """
    FeedForward module

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.
        multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        ffn_dim_multiplier (Optional[float]): Custom multiplier for hidden dimension. Defaults to None.

    Attributes:
        w1 (Linear): Linear transformation for the first layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Linear): Linear transformation for the third layer.

    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(
            self,
            init_std: float,
            residual_div: float,
            init_gate_as_residual: bool,
            init_fn_type: str,
    ):
        init_fn = build_init_fn(init_fn_type)
        init_fn(self.w1.weight, mean=0.0, std=init_std)
        init_fn(self.w2.weight, mean=0.0, std=init_std / residual_div)
        gate_init_std = (
            init_std / residual_div
            if init_gate_as_residual
            else init_std
        )
        init_fn(self.w3.weight, mean=0.0, std=gate_init_std)


class MoE(nn.Module):
    def __init__(
        self,
        dim: int,
        multiple_of: int = 256,
        n_shared_experts: int = 2,
        n_routed_experts: int = 8,
        activate_experts: int = 2,
        ffn_dim_multiplier: Optional[float] = None,
        match_dim_with_dense: bool = True,
        bias_update_speed: float = 0.01,  # Bias adjustment speed
        aux_loss_alpha: float = 0.01,  # Small weight for sequence-wise auxiliary loss
    ):
        super().__init__()
        """
        match_dim_with_dense.

        Saying the dense model have the 'hidden_size' of 1024, 
        Then "actual_dim * activate_experts = hidden_size" should be satisfied.
        """

        if match_dim_with_dense:
            total_activate_experts = n_shared_experts + activate_experts
            ratio = total_activate_experts / (n_routed_experts + n_shared_experts)
        else:
            ratio = 1.0

        hidden_dim = 4 * dim
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = int(hidden_dim * ratio)

        hidden_dim += -hidden_dim % multiple_of

        self.n_routed_experts = n_routed_experts
        self.topk = activate_experts
        self.n_shared_experts = n_shared_experts
        self.aux_loss_alpha = aux_loss_alpha  # Loss coefficient

        # Use updated Gate with DeepSeekMoE-style routing and bias balancing
        self.gate = Gate(
            hidden_size=dim,
            experts=n_routed_experts,
            topk=activate_experts,
            bias=True,
            bias_update_speed=bias_update_speed,
        )

        # Shared Experts (applies to all tokens)
        if n_shared_experts > 0:
            self.shared_experts = GroupedExperts(
                dim_in=dim, dim_out=hidden_dim, num_experts=n_shared_experts
            )

        else:
            self.shared_experts = None

        # Routed Experts (only used when selected)
        if n_routed_experts > 0:
            self.experts = GroupedExperts(
                dim_in=dim, dim_out=hidden_dim, num_experts=n_routed_experts
            )

        else:
            self.experts = None

    def init_weights(
            self,
            init_std: float,
            residual_div: float,
            init_gate_as_residual: bool,
            init_fn_type: str,
    ):
        if self.experts is not None:
            self.experts.init_weights(
                init_std,
                residual_div=residual_div,
                init_gate_as_residual=init_gate_as_residual,
                init_fn_type=init_fn_type,
            )
        if self.shared_experts is not None:
            self.shared_experts.init_weights(
                init_std,
                residual_div=residual_div,
                init_gate_as_residual=init_gate_as_residual,
                init_fn_type=init_fn_type,
            )

    def update_gate_bias(self):
        self.gate.update()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        bz, slen, dim = x.shape
        x = rearrange(x, "b s d -> (b s) d")  # Flatten batch & sequence
        out = torch.zeros_like(x)

        if self.topk > 0:
            weights, indices, scores = self.gate(
                x, update_bias=self.training
            )  # (B*S, K), (B*S, K)
            _, num_experts = scores.shape
            # routed_input shape (num_experts*tokens_per_expert, dim)
            choosen_indices = indices.reshape(-1, 1).expand(-1, dim)
            # routed_input shape (num_experts*tokens_per_expert, dim)
            routed_input = torch.gather(x, dim=0, index=choosen_indices)
            routed_input = routed_input * weights.reshape(-1, 1)
            # routed_input shape (num_experts, tokens_per_expert, dim_in)
            routed_input = routed_input.reshape(num_experts, -1, dim)
            # routed_output shape (num_experts, tokens_per_expert, dim_out)
            routed_output = self.experts(routed_input)
            # routed_output shape (num_experts*tokens_per_expert, dim_out)
            routed_output = routed_output.reshape(-1, dim)

            out = out.scatter_add(dim=0, index=choosen_indices, src=routed_output)

        # Apply shared experts if they exist
        if self.shared_experts is not None:
            # Reshape input for shared experts: (1, B*S, dim)
            shared_input = x.reshape(1, bz * slen, dim)
            # Get shared expert outputs: (1, B*S, dim)
            shared_output = self.shared_experts(shared_input)
            # Reshape back to (B*S, dim) and add to output
            shared_output = shared_output.reshape(bz * slen, dim)
            out = out + shared_output

        if self.training and self.aux_loss_alpha > 0 and self.topk > 0:
            aux_loss = self.sequence_wise_aux_loss(
                indices,
                weights,
                scores,
                bz,
                slen,
            )
        else:
            aux_loss = torch.tensor(0.0, device=x.device)

        if self.topk > 0:
            routing_entropy = -(weights * weights.log()).sum(dim=-1).mean()
        else:
            routing_entropy = torch.tensor(0.0, device=x.device)

        output = rearrange(out, "(b s) d -> b s d", b=bz, s=slen)
        return output, aux_loss, routing_entropy

    def sequence_wise_aux_loss(
        self,
        indices: torch.Tensor,
        weights: torch.Tensor,
        scores: torch.Tensor,
        B: int,
        S: int,
    ) -> torch.Tensor:
        """
        Computes the Sequence-Wise Auxiliary Loss as described in DeepSeekMoE.

        Args:
            indices (torch.Tensor): Selected expert indices (B*S, K).
            weights (torch.Tensor): Routing weights (B*S, K).
            scores (torch.Tensor): Raw expert scores before top-k selection (B*S, N_r).
            B (int): Batch size.
            S (int): Sequence length.

        Returns:
            aux_loss (torch.Tensor): Computed sequence-wise auxiliary loss.
        """
        indices = indices.view(B * S, self.topk)  # (B*S, K)
        weights = weights.view(B * S, self.topk)  # (B*S, K)
        scores = scores.view(B * S, self.n_routed_experts)  # (B*S, N_r)

        N_r = self.n_routed_experts  # Total number of routed experts
        topk = self.topk  # Number of experts per token

        # Compute expert frequency f_i (fraction of tokens assigned to expert i)
        f_i = torch.zeros(N_r, device=indices.device, dtype=weights.dtype)
        f_i.scatter_add_(
            0, indices.view(-1), torch.ones_like(indices.view(-1), dtype=weights.dtype)
        )
        f_i /= B * S * topk  # Normalize by total tokens and top-k experts

        # Compute normalized expert probabilities P_i
        sum_scores = scores.sum(
            dim=-1, keepdim=True
        )  # Sum across all experts per token (B*S, 1)
        normalized_scores = scores / sum_scores.clamp(
            min=1e-5
        )  # Avoid division by zero

        P_i = torch.zeros(N_r, device=indices.device, dtype=weights.dtype)
        for k in range(topk):
            P_i.scatter_add_(
                0,
                indices[:, k],
                normalized_scores.gather(1, indices[:, k].unsqueeze(-1)).squeeze(-1),
            )

        P_i /= B * S  # Normalize by total sequence length

        # Compute final auxiliary loss
        aux_loss = (f_i * P_i).sum() * self.aux_loss_alpha

        return aux_loss


class TransformerBlock(nn.Module):
    """
    TransformerBlock Module

    Args:
        layer_id (int): Identifier for the layer.
        model_args (TransformerModelArgs): Model configuration arguments.

    Attributes:
        n_heads (int): Number of attention heads.
        dim (int): Dimension size of the model.
        head_dim (int): Dimension size of each attention head.
        attention (Attention): Attention module.
        feed_forward (FeedForward): FeedForward module.
        layer_id (int): Identifier for the layer.
        attention_norm (RMSNorm): Layer normalization for attention output.
        ffn_norm (RMSNorm): Layer normalization for feedforward output.

    """

    attention_cls = Attention
    feed_forward_cls = MoE

    def __init__(self, layer_id: int, model_args: MoEModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim
        self.attention = self.attention_cls(model_args)
        self.feed_forward = self.feed_forward_cls(
            dim=model_args.dim,
            multiple_of=model_args.multiple_of,
            ffn_dim_multiplier=model_args.ffn_dim_multiplier,
            n_shared_experts=model_args.n_shared_experts,
            n_routed_experts=model_args.n_routed_experts,
            activate_experts=model_args.activate_experts,
            bias_update_speed=model_args.moe_gate_bias_update_speed,
            aux_loss_alpha=model_args.moe_aux_loss_alpha,
            match_dim_with_dense=True,
        )

        self.layer_id = layer_id
        self.num_layers = model_args.n_layers

        self.attention_norm = build_norm(
            model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps
        )
        self.ffn_norm = build_norm(
            model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps
        )

        self.weight_init_fn_type = model_args.intermediate_init_fn_type
        self.weight_init_std = (
            model_args.intermediate_init_std
            * model_args.dim ** model_args.intermediate_exp
        )
        if model_args.depth_init:
            self.residual_div = (2 * (self.layer_id + 1)) ** 0.5
        else:
            self.residual_div = (2 * self.num_layers) ** 0.5
        self.init_gate_as_residual = model_args.init_gate_as_residual

    def update_gate_bias(self):
        self.feed_forward.update_gate_bias()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        h = x + self.attention(self.attention_norm(x), freqs_cis)

        mlp_output, moe_aux_loss, routing_entropy = self.feed_forward(self.ffn_norm(h))

        out = h + mlp_output
        return out, moe_aux_loss, routing_entropy

    def init_weights(self):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(
            self.weight_init_std,
            residual_div=self.residual_div,
            init_fn_type=self.weight_init_fn_type,
        )
        self.feed_forward.init_weights(
            self.weight_init_std,
            residual_div=self.residual_div,
            init_gate_as_residual=self.init_gate_as_residual,
            init_fn_type=self.weight_init_fn_type,
        )


class Transformer(nn.Module, ModelProtocol):
    """
    Transformer Module

    Args:
        model_args (TransformerModelArgs): Model configuration arguments.

    Attributes:
        model_args (TransformerModelArgs): Model configuration arguments.
        vocab_size (int): Vocabulary size.
        n_layers (int): Number of layers in the model.
        tok_embeddings (ParallelEmbedding): Token embeddings.
        layers (torch.nn.ModuleList): List of Transformer blocks.
        norm (RMSNorm): Layer normalization for the model output.
        output (ColumnParallelLinear): Linear layer for final output.
        freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

    """

    transformer_block_cls = TransformerBlock

    def __init__(self, model_args: MoEModelArgs):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers

        print(
            f"model_args.dim = {model_args.dim} | model_args.vocab_size = {model_args.vocab_size}"
        )

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)

        # TODO persistent should be set to false, since this buffer can be recomputed.
        # however, we set it to true for 2 reasons.  (1) due to pytorch/pytorch#123411,
        # compile or pipeline-tracer will not correctly handle non-persistent buffers,
        # so we need to fix that.  (2) if we initialize pipeline-parallel models from
        # a seed checkpoint rather than calling init_weights, we need freqs_cis to be
        # initialized by the checkpoint, or we need to add a separate initializer for
        # just the non-persistent buffers that is called after loading checkpoints.
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=True)

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = self.transformer_block_cls(
                layer_id, model_args
            )

        self.norm = build_norm(
            model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps
        )

        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)
        self.init_weights()

    def init_weights(
        self,
        buffer_device: Optional[torch.device] = None,
    ):
        """
        [Note: On ``init_weights`` vs. ``reset_parameters``]
        Modules may define ``reset_parameters`` to initialize parameter values.
        ``reset_parameters`` is meant to only initialize directly owned
        parameters/buffers, not those of their child modules, and it can be
        used to give the initial values for these tensors.
        Separately, users may want custom initialization for their modules,
        different from that in ``reset_parameters``. For this, we define
        ``init_weights``. We only call it in the constructor of this
        ``Transformer`` root module to avoid reinitializing tensors.
        """
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            self.freqs_cis = self._precompute_freqs_cis()
        first_in_init_fn = build_init_fn(self.model_args.first_in_init_fn_type)
        first_in_std = (
            self.model_args.first_in_init_std
            * self.model_args.vocab_size ** self.model_args.first_in_exp
        )
        if self.tok_embeddings is not None:
            first_in_init_fn(
                self.tok_embeddings.weight,
                mean=0.0,
                std=first_in_std,
            )
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights()
        if self.norm is not None:
            self.norm.reset_parameters()
        final_out_init_fn = build_init_fn(self.model_args.final_out_init_fn_type)
        final_out_std = (
            self.model_args.final_out_init_std
            * self.model_args.dim ** self.model_args.final_out_exp
        )
        cutoff_factor = 3
        if self.output is not None:
            extra_kwargs = {}
            if self.init_fn_type == "trunc_normal":
                extra_kwargs["a"] = -cutoff_factor * final_out_std
                extra_kwargs["b"] = cutoff_factor * final_out_std
            final_out_init_fn(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                **extra_kwargs,
            )

    def _precompute_freqs_cis(self) -> torch.Tensor:
        return precompute_freqs_cis(
            self.model_args.dim // self.model_args.n_heads,
            # Need to compute until at least the max token limit for generation
            # TODO: explain in docs/composability.md why we removed the 2x
            # relaxing in our CP enablement PR
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )

    def update_gate_bias(self):
        for layer in self.layers.values():
            layer.update_gate_bias()

    def forward(self, inputs: MoEInputs) -> MoEInputsDict:
        """
        Perform a forward pass through the Transformer model.

        Args:
            inputs (MoEInputs): Single tensor or dictionary containing the
                following keys and values:
                - tokens_list (Union[list[Optional[torch.Tensor]],
                  torch.Tensor]): Input token indices.
                - aux_loss (torch.Tensor): Sequence-wise auxiliary balance loss.
                - moe_entropy_per_layer (torch.Tensor): Entropy of MoE routing
                  for each MoE layer.

        Returns:
            MoEInputsDict: Dictionary containing the following keys and values:
                - tokens_list (list[torch.Tensor]): Output logits after applying
                  the Transformer model.
                - aux_loss (torch.Tensor): Sequence-wise auxiliary balance loss.
                - moe_entropy_per_layer (torch.Tensor): Entropy of MoE routing
                  for each MoE layer.

        """
        if not isinstance(inputs, dict):
            inputs = {"tokens_list": inputs}
        tokens = inputs["tokens_list"]
        total_moe_aux_loss = inputs.get("aux_loss", 0)
        moe_entropy_per_layer = inputs.get("moe_entropy_per_layer", {})
        if isinstance(tokens, list):
            tokens = tokens[0]

        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        for layer in self.layers.values():
            h, moe_aux_loss, routing_entropy = layer(h, self.freqs_cis)
            total_moe_aux_loss += moe_aux_loss
            moe_entropy_per_layer[layer.layer_id] = routing_entropy.detach()

        h = self.norm(h) if self.norm else h
        output = self.output(h) if self.output else h
        return {
            "tokens_list": [output],
            "aux_loss": total_moe_aux_loss,
            "moe_entropy_per_layer": moe_entropy_per_layer,
        }

    @classmethod
    def from_model_args(cls, model_args: MoEModelArgs) -> "Transformer":
        """
        Initialize a Transformer model from a TransformerModelArgs object.

        Args:
            model_args (TransformerModelArgs): Model configuration arguments.

        Returns:
            Transformer: Transformer model.

        """
        return cls(model_args)
