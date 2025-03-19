# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

import torch

from torchtitan.config_manager import JobConfig


def cross_entropy_loss(
        pred: Union[torch.Tensor, list[torch.Tensor]],
        labels: torch.Tensor,
) -> torch.Tensor:
    """Common cross-entropy loss function for Transformer models training."""
    if isinstance(pred, list):
        pred = pred[0]
    return torch.nn.functional.cross_entropy(
        pred.flatten(0, 1).float(), labels.flatten(0, 1)
    )


# TODO: compiling loss function causes CUDA errors, turning off for now
# compiled_cross_entropy_loss = torch.compile(cross_entropy_loss)


def chunked_cross_entropy_loss(
    logits: Union[torch.Tensor, list[torch.Tensor]],
    labels: torch.Tensor,
    num_output_chunks: int = 8,
) -> torch.Tensor:
    if isinstance(logits, list):
        logits = logits[0]

    # Adapted from torchtune
    # https://github.com/pytorch/torchtune/blob/c3703482bde72e572b535d3f7c43c81e94164ebc/torchtune/modules/loss/ce_chunked_output_loss.py

    labels = [target_chunk for target_chunk in labels.chunk(num_output_chunks, dim=1)]
    logits = [logit_chunk for logit_chunk in logits.chunk(num_output_chunks, dim=1)]

    # compute one chunk at a time
    total_loss = 0.0
    for logits_chunk, labels_chunk in zip(logits, labels):
        total_loss += cross_entropy_loss(logits_chunk, labels_chunk)

    return total_loss / num_output_chunks


def multi_token_cross_entropy_loss(
        preds: list[torch.Tensor],
        labels: torch.Tensor,
        job_config: JobConfig,
) -> torch.Tensor:
    """Multi-token cross-entropy loss function for Transformer model training.

    Based on DeepSeek-V3 technical report: https://arxiv.org/abs/2412.19437.
    """
    main_loss = cross_entropy_loss(preds[0], labels[:, :job_config.training.seq_len])

    mtp_loss = 0
    for (label_offset, pred) in enumerate(preds[1:], 1):
        loss = cross_entropy_loss(
            pred,
            labels[:, label_offset:label_offset + job_config.training.seq_len],
        )
        # Take average over MTP predictions.
        loss = loss / job_config.training.num_mtp_tokens
        mtp_loss = mtp_loss + loss
    return main_loss + mtp_loss * job_config.training.mtp_loss_weight
