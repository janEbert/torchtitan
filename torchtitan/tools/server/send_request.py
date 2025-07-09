from argparse import ArgumentParser
import logging
import socket
import sys
from typing import Any

from torchtitan.datasets.tokenizer.byte_tokenizer import ByteTokenizer
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.server.serve_model import (
    decode_data,
    DEFAULT_PORT,
    encode_data,
    receive_data,
    send_data,
    TorchTitanServerRequestHandler,
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


def send_request(input_dict: dict[str, Any], server):
    # TODO asynchronously send to various servers
    logger.debug(f"Sending: {input_dict}")
    data = encode_data(input_dict)
    send_data(
        data,
        server,
        TorchTitanServerRequestHandler.MAX_SEND_DATA_BYTES,
        TorchTitanServerRequestHandler.DATA_BYTES_PER_PIECE,
    )


def main(args_list: list[str] | None = None):
    args = parse_args(args_list)

    prompts = ["This is "]
    # prompts = ["This is ", ""]
    # prompts = ["", "This is "]
    # prompts = ["This is ", "Klar doch, "]
    # prompts = ["Klar doch, ", "This is "]
    result_bytes = [[] for _ in range(len(prompts))]
    # start_pos = 0
    start_pos = -1
    # start_pos = [0]
    seed = args.seed  # 8 is nice for debugging
    seeds = []

    for i in range(10):
        input_dict = dict(input=prompts, start_pos=start_pos, seed=seed, top_k=1)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # Connect to server and send data
            sock.connect((args.server_address, args.server_port))
            send_request(input_dict, sock)

            # Receive data from the server and shut down
            received = receive_data(
                sock,
                TorchTitanServerRequestHandler.MAX_RECV_DATA_BYTES,
                TorchTitanServerRequestHandler.DATA_BYTES_PER_PIECE,
            )

        output_dict = decode_data(received)
        start_pos = output_dict["next_start_pos"]
        logger.debug(f"{output_dict = }")
        # if i == 2:
        #     break
        if "output_tokens" not in output_dict:
            raise RuntimeError(
                "server only returned logits; client-side sampling not implemented ATM",
            )
        seeds.append(output_dict["seed"])
        for (i, (prefix, suffix, input_toks)) in enumerate(zip(
                result_bytes,
                output_dict["output_tokens"],
                output_dict["input_tokens"],
        )):
            if not prefix:
                prefix = input_toks
            result_bytes[i] = prefix + suffix
            # Switch from string input to raw bytes input to handle
            # partial UTF-8.
            # TODO bottom commented one works definitely; testing full prompt with start pos now
            # This one probably has an error; isn't numerically
            # equal unlike suffix-only case.
            if start_pos < 0:
                prompts[i] = prefix + suffix
            else:
                prompts[i] = suffix

    tok = ByteTokenizer()
    print(f'{result_bytes = }')
    result_texts = [tok.decode(b) for b in result_bytes]
    # result_texts = [b.decode(errors="replace") for b in result_bytes]
    if all(s == seeds[0] for s in seeds):
        seeds_text = f"seed = {seeds[0]}"
    else:
        seeds_text = f"seeds = {seeds}"
    logger.info(f"Result texts ({seeds_text}): {result_texts}")


if __name__ == '__main__':
    init_logger()
    log_level = logging.DEBUG
    logger.setLevel(log_level)
    for handler in logger.handlers:
        handler.setLevel(log_level)
    main()
