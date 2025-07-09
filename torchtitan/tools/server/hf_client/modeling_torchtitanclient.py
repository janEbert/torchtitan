import socket
from typing import Any
import warnings

import torch
from transformers import PreTrainedModel
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import KwargsForCausalLM
from transformers.processing_utils import Unpack

from torchtitan.tools.logging import logger
from torchtitan.tools.server.hf_client.configuration_torchtitanclient import TorchTitanClientConfig
from torchtitan.tools.server.serve_model import (
    decode_data,
    encode_data,
    receive_data,
    send_data,
    TorchTitanServerRequestHandler,
)


def send_request(input_dict: dict[str, Any], server):
    logger.debug(f"Sending: {input_dict}")
    data = encode_data(input_dict)
    send_data(
        data,
        server,
        TorchTitanServerRequestHandler.MAX_SEND_DATA_BYTES,
        TorchTitanServerRequestHandler.DATA_BYTES_PER_PIECE,
    )


class TorchTitanClientCache(Cache):
    def __init__(self):
        super().__init__()
        # Cache starts enabled.
        self.reset()

    def reset(self):
        self.start_pos = 0


class TorchTitanClientPreTrainedModel(PreTrainedModel):
    config_class = TorchTitanClientConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = []
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_3 = False
    _supports_flash_attn_2 = False
    _supports_sdpa = False
    _supports_flex_attn = False
    _supports_cache_class = False
    _supports_quantized_cache = False
    _supports_static_cache = False
    _supports_attention_backend = False


class TorchTitanClientModel(TorchTitanClientPreTrainedModel):
    def __init__(self, config: TorchTitanClientConfig):
        super().__init__(config)

        self.server_address = config.server_address
        self.server_port = config.server_port
        self.seed = config.seed

        # self.vocab_size = self.model.vocab_size
        # self.padding_idx = self.model.vocab_size

        # Add a dummy parameter
        self.register_parameter(
            "dummy",
            torch.nn.Parameter(torch.tensor(0.0, requires_grad=False), requires_grad=False),
        )
        # We do _not_ want to call `self.post_init`

    def set_seed(self, seed: int | None):
        self.seed = seed

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        assert input_ids is not None, "supplying `input_ids` is required"
        assert inputs_embeds is None, "`inputs_embeds` is not supported"
        assert output_attentions is None, "`output_attentions` is not supported"
        assert output_hidden_states is None, "`output_hidden_states` is not supported"
        assert isinstance(past_key_values, (TorchTitanClientCache, type(None))), \
            "`past_key_values` has to be a `TorchTitanClientCache` or `None`."
        if attention_mask is not None:
            warnings.warn("`attention_mask` is not supported and will be ignored")
        if position_ids is not None:
            warnings.warn("`position_ids` is not supported and will be ignored")
        if cache_position is not None:
            warnings.warn("`cache_position` is not supported and will be ignored")

        if use_cache is None:
            use_cache = self.config.use_cache

        if use_cache:
            if past_key_values is None:
                past_key_values = TorchTitanClientCache()

            start_pos = past_key_values.start_pos
        else:
            start_pos = -1

        if start_pos > 0:
            assert input_ids.shape[1] == 1, \
                "currently, inputs after the KV-cache has been created need to be given one by one"

        input_dict = dict(input=input_ids.tolist(), start_pos=start_pos, seed=self.seed)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # Connect to server and send data
            sock.connect((self.server_address, self.server_port))
            send_request(input_dict, sock)

            # Receive data from the server and shut down
            received = receive_data(
                sock,
                TorchTitanServerRequestHandler.MAX_RECV_DATA_BYTES,
                TorchTitanServerRequestHandler.DATA_BYTES_PER_PIECE,
            )
        output_dict = decode_data(received)
        start_pos = output_dict["next_start_pos"]
        if use_cache:
            past_key_values.start_pos = start_pos

        return BaseModelOutputWithPast(
            last_hidden_state=torch.tensor(output_dict["output_logits"]),
            past_key_values=past_key_values if use_cache else None,
            hidden_states=None,
            attentions=None,
        )


class TorchTitanClientForCausalLM(TorchTitanClientPreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.model = TorchTitanClientModel(config)
        # self.vocab_size = self.model.vocab_size

        # We do _not_ want to call `self.post_init`

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        # `last_hidden_state` already contains the logits.
        logits = outputs.last_hidden_state
        # Only compute necessary logits
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = logits[:, slice_indices, :]

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
