from argparse import ArgumentParser
import os
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from torchtitan.tools.server.hf_client import (
    TorchTitanByteTokenizer,
    TorchTitanClientConfig,
    TorchTitanClientForCausalLM,
)
from torchtitan.tools.server.serve_model import (
    DEFAULT_PORT,
    logits_to_probs,
    multinomial_sample_one,
)


def parse_args(args_list: list[str] | None = None):
    parser = ArgumentParser()

    parser.add_argument(
        "--server_address",
        default="localhost",
        help="Address to query the server from.",
    )
    parser.add_argument(
        "--server_port",
        default=DEFAULT_PORT,
        type=int,
        help="Port to query the server from.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed to initialize the server's sampling random number generator from.",
    )

    if args_list is None:
        args_list = sys.argv[1:]
    args = parser.parse_args(args_list)
    return args


def main(args_list: list[str] | None = None):
    args = parse_args(args_list)

    use_auto = True
    use_generate = True
    seed = args.seed

    if use_auto:
        hf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./hf_client"))
        config = AutoConfig.from_pretrained(hf_path, trust_remote_code=True)
        config.server_address = args.server_address
        config.server_port = args.server_port
        config.seed = args.seed
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        # model = AutoModelForCausalLM.from_pretrained(
        #     hf_path,
        #     server_address=args.server_address,
        #     server_port=args.server_port,
        #     trust_remote_code=True,
        # )
        tok = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
        # model.save_pretrained(hf_path + "_new")
        # tok.save_pretrained(hf_path + "_new")
    else:
        config = TorchTitanClientConfig(
            server_address=args.server_address,
            server_port=args.server_port,
            seed=args.seed,
        )
        tok = TorchTitanByteTokenizer()
        model = TorchTitanClientForCausalLM(config)

    prompts = ["This is "]
    encoded = tok(prompts, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"]
    # attention_masks = encoded["attention_mask"]
    if not use_generate:
        outputs = model(input_ids)

        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        else:
            rng.seed()
        seed = rng.initial_seed()

        output_probs = logits_to_probs(outputs["logits"][:, -1, :])
        output_ids = multinomial_sample_one(output_probs, rng=rng)

        responses_encoded = torch.cat([input_ids, output_ids], dim=1)
    else:
        old_rng_state = torch.get_rng_state()
        if seed is not None:
            torch.manual_seed(seed)
        seed = torch.initial_seed()

        responses_encoded = model.generate(input_ids, max_new_tokens=20, do_sample=True)

        torch.set_rng_state(old_rng_state)
    # print(f'{responses_encoded = }')
    responses = tok.batch_decode(responses_encoded, skip_special_tokens=True)
    # responses = tok.batch_decode(responses_encoded, skip_special_tokens=False)
    print(f'{seed = }')
    print(f'{responses = }')


if __name__ == "__main__":
    main()
