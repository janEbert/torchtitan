import json
import os
import socket
from typing import Any
import warnings

import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import KwargsForCausalLM
from transformers.processing_utils import Unpack
from transformers.tokenization_utils import PreTrainedTokenizer

from torchtitan.datasets.tokenizer.byte_tokenizer import ByteTokenizer
from torchtitan.tools.logging import logger
from torchtitan.tools.server.serve_model import (
    decode_data,
    DEFAULT_PORT,
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


class TorchTitanClientConfig(PretrainedConfig):
    model_type = "torchtitan_client"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
            self,
            server_address: str = "localhost",
            server_port: int | None = None,
            use_cache: bool = True,
            seed: int | None = None,
            **kwargs,
    ):
        self.server_address = server_address
        if server_port is None:
            server_port = DEFAULT_PORT
        self.server_port = server_port
        self.use_cache = use_cache
        self.seed = seed
        super().__init__(**kwargs)


class TorchTitanClientCache(Cache):
    def __init__(self):
        super().__init__()
        # Cache starts enabled.
        self.reset()

    def reset(self):
        self.start_pos = 0


class TorchTitanTokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids", "attention_mask"]
    padding_side = "left"
    truncation_side = "left"


class TorchTitanByteTokenizer(TorchTitanTokenizer):
    def __init__(
        self,
        bos_token: str | None = None,
        eos_token: str | None = None,
        pad_token: str | None = None,
        additional_special_tokens: tuple[str] | list[str] | None = None,
        add_bos_token: bool = True,
        add_eos_token: bool = False,
    ):
        tok_init_kwargs = {}
        if bos_token is not None:
            tok_init_kwargs["bos_token"] = bos_token
        if eos_token is not None:
            tok_init_kwargs["eos_token"] = eos_token
        if pad_token is not None:
            tok_init_kwargs["pad_token"] = pad_token
        if isinstance(additional_special_tokens, tuple):
            additional_special_tokens = list(additional_special_tokens)

        tok_init_kwargs["special_tokens"] = additional_special_tokens
        self.tokenizer = ByteTokenizer(**tok_init_kwargs)

        super_init_kwargs = {}
        # Update special tokens with tokenizer settings.
        super_init_kwargs["bos_token"] = self.tokenizer.inv_vocab[self.tokenizer.bos_id]
        super_init_kwargs["eos_token"] = self.tokenizer.inv_vocab[self.tokenizer.eos_id]
        super_init_kwargs["pad_token"] = self.tokenizer.inv_vocab[self.tokenizer.pad_id]
        additional_special_tokens = self.tokenizer.special_tokens.copy()
        additional_special_tokens.pop(super_init_kwargs["bos_token"], None)
        additional_special_tokens.pop(super_init_kwargs["eos_token"], None)
        additional_special_tokens.pop(super_init_kwargs["pad_token"], None)
        super_init_kwargs["additional_special_tokens"] = list(additional_special_tokens.keys())
        super().__init__(**super_init_kwargs)

        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.n_words

    def get_vocab(self) -> dict[str, int]:
        return self.tokenizer.vocab

    def _tokenize(self, text, **kwargs) -> list[str]:
        return list(text)

    # # Hack `self._tokenize` to be able to pass `bos` or `eos` to it.
    # def tokenize(self, *args, **kwargs) -> list[str]:
    #     bos = kwargs.pop("bos", self.add_bos_token)
    #     eos = kwargs.pop("eos", self.add_eos_token)
    #     old_tokenize = self._tokenize

    #     def new_tokenize(text, **kwargs):
    #         return old_tokenize(text, bos=bos, eos=eos, **kwargs)

    #     self._tokenize = new_tokenize
    #     tokens = super().tokenize(*args, **kwargs)
    #     self._tokenize = old_tokenize
    #     return tokens

    # def _tokenize(self, text, **kwargs) -> list[str]:
    #     bos = kwargs.get("bos", self.add_bos_token)
    #     eos = kwargs.get("eos", self.add_eos_token)
    #     return self.tokenizer.encode(text, bos=bos, eos=eos)

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
    ) -> list[int]:
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output

    def _convert_token_to_id(self, token: str) -> int:
        return self.tokenizer.vocab[token]

    def _convert_id_to_token(self, index: int) -> str:
        return self.tokenizer.inv_vocab[index]

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        text = "".join(tokens)
        return text
        # token_ids = self.tokenizer.encode(text, bos=False, eos=False)
        # return self.tokenizer.decode(token_ids)

    def save_vocabulary(
            self,
            save_directory: str,
            filename_prefix: str | None = None,
    ) -> tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "byte-tok-vocab.json",
        )
        with open(out_vocab_file, "w") as f:
            json.dump(self.tokenizer.vocab, f, indent=4)
        return (out_vocab_file,)


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
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = logits[:, slice_indices, :]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
