# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from typing import Callable, ClassVar, Optional
import warnings

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)


class FlexAttention(torch.nn.Module):
    # We registered flex_attention related attributes as class variables as we
    # need to amortize the cost of compilation.
    flex_attn: ClassVar[Callable] = torch.compile(
        flex_attention, mode="max-autotune-no-cudagraphs"
    )
    compiled_create_block_mask: ClassVar[Callable] = torch.compile(create_block_mask)
    used_attn_mask_types: ClassVar[set[str]] = set()
    # Attention mask type to the created BlockMask.
    # This allows us to keep track the created block masks for each
    # new batch. We will use this to update the block mask when a
    # new batch is created. This also allows user to create different
    # block masks for different layers.
    block_masks: ClassVar[dict[tuple[str, int], BlockMask]] = {}
    pad_cache: ClassVar[dict[str, torch.Tensor]] = {}

    # Instance variables.
    attn_mask_type: str

    def __init__(self, attn_mask_type: str) -> None:
        super().__init__()
        if attn_mask_type not in ["causal", "block_causal"]:
            raise ValueError(f"Unrecognized attn_mask_type {attn_mask_type}.")
        self.attn_mask_type = attn_mask_type
        FlexAttention.used_attn_mask_types.add(attn_mask_type)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        seq_len = q.shape[2]
        block_mask = FlexAttention.block_masks[(self.attn_mask_type, seq_len)]
        return FlexAttention.flex_attn(q, k, v, block_mask=block_mask)

    @staticmethod
    def _get_causal_mask_fn() -> Callable:
        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        return causal_mask

    @staticmethod
    def _get_padded_causal_mask_fn(
            batch: torch.Tensor,
            start_pos: int,
            pad_id: int,
            pad_cache: torch.Tensor | None = None,
    ) -> Callable:
        # batch is [b, s, h, d] shape
        mask = batch != pad_id
        q_offset = 0
        if start_pos > 0:
            if pad_cache is None:
                warnings.warn(
                    "Padding with KV caching is used, but pad cache was not initialized. "
                    "Initializing automatically, but this may incorrect wrong results."
                )
                pad_cache = torch.ones(
                    (mask.shape[0], start_pos + 1 - mask.shape[1]) + mask.shape[2:],
                    dtype=mask.dtype,
                    device=mask.device,
                )
            mask = torch.hstack([pad_cache, mask])
            # If we have a partial input, offset queries.
            if batch.shape[1] <= start_pos:
                q_offset = start_pos
        if start_pos >= 0:
            pad_cache = mask

        def padded_causal_mask(b, h, q_idx, kv_idx):
            return mask[b, q_idx + q_offset] & mask[b, kv_idx] & (q_idx + q_offset >= kv_idx)

        return padded_causal_mask, pad_cache

    @staticmethod
    def _get_cached_causal_mask_fn(batch: torch.Tensor, start_pos: int) -> Callable:
        assert start_pos >= 0

        q_offset = 0
        # If we have a partial input, offset queries.
        if batch.shape[1] <= start_pos:
            q_offset = start_pos

        def cached_causal_mask(b, h, q_idx, kv_idx):
            return (q_idx + q_offset >= start_pos) & (q_idx + q_offset >= kv_idx)

        return cached_causal_mask

    @staticmethod
    def _get_cached_padded_causal_mask_fn(
            batch: torch.Tensor,
            start_pos: int,
            pad_id: int,
            pad_cache: torch.Tensor | None = None,
    ) -> Callable:
        assert start_pos >= 0

        # batch is [b, s, h, d] shape
        mask = batch != pad_id
        q_offset = 0
        if start_pos > 0:
            if pad_cache is None:
                warnings.warn(
                    "Padding with KV caching is used, but pad cache was not initialized. "
                    "Initializing automatically, but this may incorrect wrong results."
                )
                pad_cache = torch.ones(
                    (mask.shape[0], start_pos + 1 - mask.shape[1]) + mask.shape[2:],
                    dtype=mask.dtype,
                    device=mask.device,
                )
            mask = torch.hstack([pad_cache, mask])
            # If we have a partial input, offset queries.
            if batch.shape[1] <= start_pos:
                q_offset = start_pos
        if start_pos >= 0:
            pad_cache = mask

        def cached_padded_causal_mask(b, h, q_idx, kv_idx):
            return (
                mask[b, q_idx + q_offset]
                & mask[b, kv_idx]
                & (q_idx + q_offset >= start_pos)
                & (q_idx + q_offset >= kv_idx)
            )

        return cached_padded_causal_mask, pad_cache

    @staticmethod
    def _get_block_causal_mask_fn(batch: torch.Tensor, eos_id: int) -> Callable:
        # batch is [b, s, h, d] shape
        mask = batch == eos_id
        mask[:, -1] = True
        acc_mask = torch.cumsum(torch.where(mask, 1, 0), dim=1)
        seq_idx = torch.zeros_like(acc_mask, dtype=torch.int32)
        seq_idx[:, 1:] = acc_mask[:, :-1]

        def block_causal_mask(b, h, q_idx, kv_idx):
            return (seq_idx[b, q_idx] == seq_idx[b, kv_idx]) & (q_idx >= kv_idx)

        return block_causal_mask

    @staticmethod
    @torch.no_grad()
    def init_attention_mask(
            batch: torch.Tensor,
            eos_id: Optional[int] = None,
            pad_id: Optional[int] = None,
            start_pos: int = -1,
    ) -> None:
        # batch is [b, s, h, d] shape
        for attn_mask_type in FlexAttention.used_attn_mask_types:
            seq_len = batch.shape[1]
            use_cached_mask = start_pos >= 0
            use_padded_mask = pad_id is not None
            if use_padded_mask:
                pad_cache = FlexAttention.pad_cache.get(attn_mask_type)
            else:
                pad_cache = None

            match attn_mask_type:
                case "causal":
                    if (
                            not use_cached_mask
                            and not use_padded_mask
                            and FlexAttention.block_masks.get((attn_mask_type, seq_len)) is not None
                    ):
                        continue
                    # We don't care about batch dimension --
                    # all samples have the same lower triangle mask.
                    batch_dimension = 1
                    if use_cached_mask and start_pos > 0:
                        if use_padded_mask:
                            batch_dimension = batch.shape[0]
                            mask_fn, pad_cache = FlexAttention._get_cached_padded_causal_mask_fn(
                                batch,
                                start_pos,
                                pad_id,
                                pad_cache,
                            )
                        else:
                            mask_fn = FlexAttention._get_cached_causal_mask_fn(batch, start_pos)
                    else:
                        if use_padded_mask:
                            batch_dimension = batch.shape[0]
                            mask_fn, pad_cache = FlexAttention._get_padded_causal_mask_fn(
                                batch,
                                start_pos,
                                pad_id,
                                pad_cache,
                            )
                        else:
                            mask_fn = FlexAttention._get_causal_mask_fn()
                case "block_causal":
                    if eos_id is None:
                        raise RuntimeError(
                            "eos_id must be provided for block_causal mask."
                        )
                    if start_pos >= 0:
                        raise ValueError('`start_pos` not supported with "block_causal" mask.')
                    if use_padded_mask:
                        warnings.warn(
                            '`pad_id` not supported with "block_causal" mask. If you are not using '
                            '`pad_id` concurrently with "block_causal" masking, it is fine to '
                            'ignore this warning.'
                        )
                    batch_dimension = batch.shape[0]
                    mask_fn = FlexAttention._get_block_causal_mask_fn(batch, eos_id)
                case _:
                    raise RuntimeError(f"Shouldn't reach here. {attn_mask_type}")

            block_mask = FlexAttention.compiled_create_block_mask(
                mask_fn,
                batch_dimension,
                None,
                seq_len,
                seq_len + max(start_pos, 0),
            )
            FlexAttention.block_masks[(attn_mask_type, seq_len)] = block_mask
            if use_padded_mask:
                FlexAttention.pad_cache[attn_mask_type] = pad_cache


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, attn_mask_type: str) -> None:
        super().__init__()
        if attn_mask_type != "causal":
            raise ValueError(
                "TorchTitan with SDPA currently only supports causal mask."
            )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def build_attention(use_flex_attn: bool, attn_mask_type: str):
    if use_flex_attn:
        return FlexAttention(attn_mask_type)
    else:
        return ScaledDotProductAttention(attn_mask_type)


def init_attention_mask(
        batch: torch.Tensor,
        eos_id: Optional[int] = None,
        pad_id: Optional[int] = None,
        start_pos: int = -1,
) -> None:
    FlexAttention.init_attention_mask(batch, eos_id, pad_id=pad_id, start_pos=start_pos)
