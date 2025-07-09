from argparse import ArgumentParser
import logging
import math
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

    result_logits = {}

    # for (orig_prompts, res_indices) in zip([
    #         ["This is " * 200],
    #         ["This is " * 200, ""],
    #         ["", "This is " * 200],
    #         # TODO these are wrong because of freq_cis not being shifted correctly.
    #         # TODO also pad token attention mask handling needs to be cached
    #         ["This is " * 200, "Klar doch, " * 180],
    #         ["Klar doch, " * 180, "This is " * 200],
    #         ["This is " * 200, "This is " * 200],
    #         ["This is " * 200, "This is " * 200, "This is " * 200],
    #         ["This is " * 200, "This is " * 200, "This is " * 200, "This is " * 200],
    # ], [(0,), (0,), (1,), (0,), (1,), (0, 1), (0, 1, 2), (0, 1, 2, 3)]):
    for (orig_prompts, res_indices) in zip([
            ["This is "],
            ["This is ", ""],
            ["", "This is "],
            # TODO these are wrong because of freq_cis not being shifted correctly.
            # TODO also pad token attention mask handling needs to be cached
            ["This is ", "Klar doch, "],
            ["Klar doch, ", "This is "],
            ["This is ", "Klar doch, " * 180],
            ["Klar doch, " * 180, "This is "],
            ["This is ", "This is "],
            ["This is ", "This is ", "This is "],
            ["This is ", "This is ", "This is ", "This is "],
    ], [(0,), (0,), (1,), (0,), (1,), (0,), (1,), (0, 1), (0, 1, 2), (0, 1, 2, 3)]):
        for start_pos in [-1, 0]:
            prompts = orig_prompts.copy()
            # if prompts == ["This is "] and start_pos == -1:
            #     continue

            orig_start_pos = start_pos
            result_bytes = [[] for _ in range(len(prompts))]
            seed = args.seed  # 8 is nice for debugging
            seeds = []

            input_dict = dict(input=prompts, start_pos=start_pos, seed=seed, top_k=1)

            for i in range(10):
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
                # TODO uncomment below
                # logger.debug(f"{output_dict = }")
                if True:
                    logger.debug(f"{output_dict = }")
                    curr_logits = []
                    for res_index in res_indices:
                        logits = output_dict["output_logits"][res_index][-1:]
                        # if not math.isclose(logits[0][0], -1.5577514171600342):
                        #     print('FUCK')
                        #     exit()
                        # if len(logits) == 1 and isinstance(logits[0], list):
                        #     logits = logits[0]
                        for logits_entry in logits:
                            curr_logits.append(dict(res_index=res_index, logits=logits_entry))
                    result_logits.setdefault(i, []).append(dict(
                        curr_logits=curr_logits,
                        prompts=orig_prompts,
                        start_pos=orig_start_pos,
                    ))
                if i == 2:
                    break
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
                    print(f'{i = }, {prompts[i] = }, {prefix = }, {suffix = }')

            tok = ByteTokenizer()
            print(f'{result_bytes = }')
            result_texts = [tok.decode(b) for b in result_bytes]
            # result_texts = [b.decode(errors="replace") for b in result_bytes]
            if all(s == seeds[0] for s in seeds):
                seeds_text = f"seed = {seeds[0]}"
            else:
                seeds_text = f"seeds = {seeds}"
            logger.info(f"Result texts ({seeds_text}): {result_texts}")

    print("comparing...")
    # print(result_logits)
    for seq_index in range(3):
        cmp_logits = None
        for (prompt_index, curr_logits_dict) in enumerate(result_logits[seq_index]):
            for (curr_logits_index, logits_dict) in enumerate(
                    curr_logits_dict["curr_logits"],
            ):
                logits = logits_dict["logits"]
                if cmp_logits is None:
                    cmp_logits = logits
                    continue

                curr_logits_dict = curr_logits_dict.copy()
                curr_logits_dict.pop("curr_logits", None)
                logits_dict = logits_dict.copy()
                logits_dict.pop("logits")
                for (i, (l, cl)) in enumerate(zip(logits, cmp_logits)):
                    # if not math.isclose(l, cl):
                    #     print(f'exact mismatch {i = }, {l} != {cl}')
                    if not math.isclose(l, cl, rel_tol=1e-4):
                        print(f'{seq_index = }, {i = }, {l} != {cl}')
                        print(f"discrepancy at {curr_logits_dict = }, {logits_dict = }")
                        break
                else:
                    print(f"no discrepancy at {seq_index = }, {curr_logits_dict = }, {logits_dict = }")


if __name__ == '__main__':
    init_logger()
    log_level = logging.DEBUG
    logger.setLevel(log_level)
    for handler in logger.handlers:
        handler.setLevel(log_level)
    main()
