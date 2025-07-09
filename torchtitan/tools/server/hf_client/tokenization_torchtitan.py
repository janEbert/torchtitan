import json
import os

from transformers.tokenization_utils import PreTrainedTokenizer

from torchtitan.datasets.tokenizer.byte_tokenizer import ByteTokenizer
from torchtitan.tools.logging import logger


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
        **kwargs,
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
