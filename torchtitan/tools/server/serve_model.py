import atexit
from argparse import ArgumentParser
import base64
import contextlib
from datetime import timedelta
import functools
import importlib
import itertools
import json
import logging
import os
import re
from socketserver import BaseRequestHandler, TCPServer
import sys
import time
from typing import Any
import warnings
import zlib

import torch
from torch._inductor.utils import is_gpu
import torch.distributed.checkpoint as dcp
from torch.distributed.pipelining import schedules
from torch.utils._triton import has_triton

from torchtitan.components.activation_offload import get_act_offloading_ctx_manager
from torchtitan.components.checkpoint import MODEL as STATE_MODEL_KEY, ModelWrapper
from torchtitan.config_manager import JobConfig
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.components.metrics import build_metrics_processor
from torchtitan.protocols.model_converter import build_model_converters
import torchtitan.protocols.train_spec as train_spec_module
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools import utils

DEFAULT_PORT = 29302


def parse_args(args_list: list[str] | None = None):
    parser = ArgumentParser()
    parser.add_argument(
        "--dump_folder",
        help=(
            "Path to the dump folder of the job; giving "
            "`checkpoint_folder` or `job_config_file` overrides locations "
            "given by this."
        ),
    )
    parser.add_argument(
        "--checkpoint_folder",
        help="Path to the folder containing the checkpoint files to convert.",
    )
    parser.add_argument(
        "--job_config_file",
        help="Path to the JSON file containing the job configuration.",
    )
    # parser.add_argument(
    #     "--model_args_file",
    #     help="Path to the JSON file containing the model arguments.",
    # )

    parser.add_argument(
        "--server_address",
        default="localhost",
        help="Address to host the server on.",
    )
    parser.add_argument(
        "--server_port",
        default=DEFAULT_PORT,
        type=int,
        nargs='*',
        help="Port to host the server on.",
    )
    parser.add_argument(
        "--do_sampling",
        action="store_true",
        help="Whether to sample instead of serving only raw logits.",
    )

    if args_list is None:
        args_list = sys.argv[1:]
    args = parser.parse_args(args_list)
    return args


@contextlib.contextmanager
def force_pipeline_schedule(
        schedule_class: schedules.PipelineScheduleSingle | schedules.PipelineScheduleMulti,
):
    old_get_schedule_class = schedules.get_schedule_class

    @functools.wraps(old_get_schedule_class)
    def new_get_schedule_class(*args, **kwargs):
        return schedule_class

    try:
        schedules.get_schedule_class = new_get_schedule_class
        yield
    finally:
        schedules.get_schedule_class = old_get_schedule_class


# From `torchtitan/scripts/generate/_generation.py`
def multinomial_sample_one(
    probs: torch.Tensor, rng: torch.Generator | None = None
) -> torch.Tensor:
    q = torch.empty_like(probs).exponential_(1, generator=rng)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.long)


# From `torchtitan/scripts/generate/_generation.py`
def logits_to_probs(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
        pivot = v.select(dim=-1, index=-1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def decode_data(data: bytes):
    data = base64.b64decode(data)
    data = zlib.decompress(data)
    data = json.loads(data)
    return data


def encode_data(data):
    data = json.dumps(data, separators=(',', ':'))
    data = data.encode()
    data = zlib.compress(data)
    data = base64.b64encode(data)
    return data


def receive_data(request, max_recv_bytes, bytes_per_piece):
    pieces = [b""]
    total_recv = 0
    while not pieces[-1].endswith(b"\n") and total_recv < max_recv_bytes:
        pieces.append(request.recv(bytes_per_piece))
        total_recv += len(pieces[-1])
    data = b"".join(pieces)
    logger.debug(f"received: {data}")
    return data


def send_data(data, request, max_send_bytes, bytes_per_piece):
    if len(data) > max_send_bytes:
        logger.warning(
            f"truncating data to send from {len(data)} bytes to "
            f"{max_send_bytes} bytes "
            f"(-{len(data) - max_send_bytes} bytes)"
        )
        data = data[:max_send_bytes]
    pieces = map(bytes, itertools.batched(data, n=bytes_per_piece))
    try:
        for piece in pieces:
            request.send(piece)
        logger.debug(f"sent: {data}")
        request.send(b"\n")
        return True
    except BrokenPipeError:
        return False


class TorchTitanServerRequestHandler(BaseRequestHandler):
    MAX_RECV_DATA_BYTES = 8_388_608  # 2^23
    MAX_SEND_DATA_BYTES = 8_388_608  # 2^23
    DATA_BYTES_PER_PIECE = 8192  # 2^13

    def handle(self):
        data = receive_data(self.request, self.MAX_RECV_DATA_BYTES, self.DATA_BYTES_PER_PIECE)
        input_dict = decode_data(data)
        logger.debug(f"Received request from {self.client_address[0]}.")
        logger.debug(f"{self.client_address[0]}: {input_dict}")

        output_dict = self.serve_step(input_dict, logits_only=self.server.logits_only)

        if output_dict is not None:
            data = output_dict
            logger.debug(f"sending: {data}")
            data = encode_data(data)
            send_data(data, self.request, self.MAX_SEND_DATA_BYTES, self.DATA_BYTES_PER_PIECE)
        # else:
        #     do nothing; this rank's output has been all-gathered

    # TODO spread/broadcast/shard/duplicate inputs depending on their batch size and DP size
    def _distribute_inputs(self, input_dict):
        inputs = input_dict["input"]
        start_pos = input_dict.get("start_pos", 0)
        assert not isinstance(start_pos, torch.Tensor), \
            'multiple start positions currently not supported'
        sharded_input_dict = input_dict.copy()

        inputs = self._normalize_inputs(inputs)

        # longtensor start pos START
        # start_pos = self._normalize_start_pos(start_pos, len(inputs))
        # assert len(start_pos) == len(inputs)
        # TODO not sharded
        # sharded_input_dict["start_pos"] = start_pos
        # longtensor start pos END

        inputs, first_seq_elem_indices, orig_seq_lens = self._to_token_tensor(inputs, start_pos)
        # TODO not sharded
        sharded_input_dict["input"] = inputs

        # Move to device
        self._to_device(sharded_input_dict, self.server.device)

        # TODO not sharded
        sharded_input_dict["first_seq_elem_indices"] = first_seq_elem_indices

        # List of indices that contain "actual" outputs, i.e., outputs
        # that were not duplicated.
        output_indices = []

        pass  # TODO

        return sharded_input_dict, output_indices

    # TODO only select outputs that this process needs
    def _select_outputs(self, output_dict, output_indices):
        pass  # TODO
        return output_dict

    def _normalize_inputs(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        return inputs

    def _normalize_start_pos(self, start_pos, batch_size):
        if isinstance(start_pos, int):
            start_pos = [start_pos] * batch_size
        if isinstance(start_pos, list):
            if len(start_pos) == 1 and len(start_pos) != batch_size:
                start_pos = start_pos * batch_size
            start_pos = torch.LongTensor(start_pos)
        return start_pos

    def _to_token_tensor(self, inputs, start_pos):
        max_input_len = -1
        if isinstance(inputs, list) and isinstance(inputs[0], str):
            assert self.server.tokenizer is not None
            tokens = []
            # longtensor start pos START
            # for (sample_text, sample_start_pos) in zip(inputs, start_pos):
            # longtensor start pos END
            for sample_text in inputs:
                sample_tokens = self.server.tokenizer.encode(
                    sample_text,
                    # Only insert BOS token when starting a new prediction.
                    bos=start_pos <= 0,
                    # longtensor start pos START
                    # bos=sample_start_pos <= 0,
                    # longtensor start pos END
                    # Never insert EOS token.
                    eos=False,
                )
                if len(sample_tokens) > max_input_len:
                    max_input_len = len(sample_tokens)
                tokens.append(sample_tokens)
            inputs = tokens
            # TODO remove anything after EOS
        # Sequence lengths before padding
        orig_seq_lens = [len(sample_tokens) for sample_tokens in inputs]
        if (
                isinstance(inputs, list)
                and isinstance(inputs[0], list)
                and isinstance(inputs[0][0], int)
        ):
            if max_input_len < 0:
                for sample_tokens in inputs:
                    if len(sample_tokens) > max_input_len:
                        max_input_len = len(sample_tokens)
            # CP seems to be handled correctly by this because the inputs
            # for it are not sharded
            # TODO is it possible that we pad inside the sequence when
            #      using CP? need to avoid this
            tokens = [
                [self.server.tokenizer.pad_id] * (max_input_len - len(sample_tokens))
                + sample_tokens
                for sample_tokens in inputs
            ]
            tokens = [
                torch.LongTensor(sample_tokens)
                for sample_tokens in tokens
            ]
            inputs = torch.stack(tokens)
        first_seq_elem_indices = max_input_len - torch.LongTensor(orig_seq_lens)
        return inputs, first_seq_elem_indices, orig_seq_lens

    def _to_device(self, input_dict, device):
        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                input_dict[k] = input_dict[k].to(device)
        return input_dict

    def serve_step(self, input_dict: dict[str, Any], logits_only: bool):
        input_dict, output_indices = self._distribute_inputs(input_dict)

        inputs = input_dict["input"]
        start_pos = input_dict.get("start_pos", 0)
        assert not isinstance(start_pos, (torch.Tensor, list)), \
            'multiple start positions currently not supported'
        sampling_params = dict(
            seed=input_dict.pop("seed", None),
            temperature=input_dict.pop("temperature", 1.0),
            top_k=input_dict.pop("top_k", None),
        )
        first_seq_elem_indices = input_dict.pop("first_seq_elem_indices")

        max_seq_len = self.server.model_parts[0].model_args.max_seq_len
        # CP seems to be handled correctly by this because the inputs
        # for it are not sharded
        for sample in inputs:
            assert len(sample) <= max_seq_len, \
                f"max sequence length exceeded ({len(sample)} > {max_seq_len})"

        # TODO last elem seq indices doesn't work with scalar start_pos
        # TODO scatter inputs across DP processes/give dummy inputs
        #      (depending on batch size); gather inputs for non-DP
        #      procs.
        #      alternatively, do the broadcasting at TCP time
        #      need to filter/remove dummy inputs either way
        logger.debug(f"{input_dict = }")
        outputs, first_seq_elem_indices = self.server.forward_and_gather(
            input_dict,
            first_seq_elem_indices,
        )
        logger.debug(f"{outputs = }")

        if outputs is not None:
            # Update start position to make it easy on users.
            # TODO with the `+ len(outputs) - 1` terms, we want to handle
            #      MTP; however, does KV caching work with it? probably
            #      depends where we start to generate from and how (i.e.,
            #      how do MTP modules handle it).
            #      probably it works correctly if just setting start_pos
            #      as if without MTP...
            # next_start_pos = start_pos + input_dict["input"].size(1) + len(output_texts[0]) - 1
            # TODO next start pos has to be minimal input seq len!
            if isinstance(start_pos, int) and start_pos == 0:
                # next_start_pos = input_dict["input"].size(1) + outputs.size(1) - 1
                # next_start_pos = last_seq_elem_indices.min().item() + 1
                next_start_pos = input_dict["input"].size(1)
            # longtensor start pos START
            # elif (
            #         isinstance(start_pos, torch.Tensor)
            #         and start_pos.dtype == torch.long
            #         and torch.all(start_pos == 0)
            # ):
            #     next_start_pos = orig_seq_lens
            # longtensor start pos END
            elif start_pos != -1:
                # next_start_pos = start_pos + outputs.size(1)
                next_start_pos = start_pos + 1
            else:
                next_start_pos = start_pos
            # longtensor start pos START
            # if isinstance(next_start_pos, torch.Tensor):
            #     next_start_pos = next_start_pos.tolist()
            # longtensor start pos END

            logger.debug(f"logits shape: {outputs.shape}")
            # Calculate probabilities
            outputs = outputs[:, -1:, :self.server.tokenizer.n_words]
            logger.debug(f"filtered logits shape: {outputs.shape}")

            # Optionally de-tokenize
            if not logits_only:
                seed = sampling_params["seed"]
                rng = torch.Generator()
                if seed is not None:
                    rng.manual_seed(seed)
                seed = rng.initial_seed()
                sampling_params["seed"] = seed

                output_logits = outputs.tolist()
                output_probs = logits_to_probs(
                    outputs[:, -1, :],
                    temperature=sampling_params["temperature"],
                    top_k=sampling_params["top_k"],
                )
                logger.debug(f"probs shape: {output_probs.shape}")
                # Sample token IDs
                outputs = multinomial_sample_one(output_probs, rng=rng)
                logger.debug(f"output token ID shape: {outputs.shape}")
                outputs = outputs.tolist()
                logger.debug(f"output bytes: {outputs}")
                output_tokens = outputs
                output_texts = []
                for output in outputs:
                    output_text = self.server.tokenizer.decode(output)
                    output_texts.append(output_text)

                output_dict = dict(
                    output_tokens=output_tokens,
                    output_probs=output_probs.tolist(),
                    output_logits=output_logits,
                    output_texts=output_texts,
                    seed=sampling_params["seed"],
                )
            else:
                output_dict = dict(
                    output_logits=outputs.tolist(),
                )

            input_tokens = input_dict["input"].tolist()
            # Remove padding
            for (i, (sample, sample_first_seq_elem_index)) in enumerate(zip(
                    input_tokens,
                    first_seq_elem_indices,
            )):
                input_tokens[i] = sample[sample_first_seq_elem_index:]
            # TODO remove anything after EOS

            output_dict.update(dict(
                next_start_pos=next_start_pos,
                input_tokens=input_tokens,
            ))

            output_dict = self._select_outputs(output_dict, output_indices)
        else:
            output_dict = None

        return output_dict


class TorchTitanServer(TCPServer):
    def __init__(self, job_config: JobConfig, checkpoint_folder: str | None = None):
        super().__init__(
            ("localhost", DEFAULT_PORT),
            TorchTitanServerRequestHandler,
            bind_and_activate=False,
        )
        self.init_model(job_config)
        if checkpoint_folder:
            self.load_model(checkpoint_folder)
        self.is_serving = False
        self.logits_only = None

    def start_server(self, address: str, port: int):
        assert not self.is_serving
        self.is_serving = True

        endpoint = (address, port)
        self.server_address = endpoint
        atexit.register(self.server_close)
        self.server_bind()
        self.server_activate()

    def serve(
            self,
            address: str,
            port: list[int] | int,
            logits_only: bool = False,
            poll_interval: float = 0.5,
    ):
        assert self.logits_only is None or self.logits_only == logits_only
        self.logits_only = logits_only

        # if not isinstance(port, list):
        #     port = [port]

        # if (
        #     self.parallel_dims.dp_replicate_enabled
        #     or self.parallel_dims.dp_shard_enabled
        #     or self.parallel_dims.cp_enabled
        # ):
        #     dp_cp_size = self.get_group_size("dp_cp")
        #     dp_cp_rank = self.get_group_rank("dp_cp")
        # else:
        #     dp_cp_size = 1
        #     dp_cp_rank = 0

        # if self.parallel_dims.dp_enabled:
        #     dp_size = self.get_group_size("dp")
        # else:
        #     dp_size = 1

        # if self.parallel_dims.cp_enabled:
        #     cp_size = self.get_group_size("cp")
        # else:
        #     cp_size = 1
        # # Sanity check
        # assert dp_size * cp_size == dp_cp_size

        # assert len(port) == 1 or len(port) == dp_cp_size, (
        #     f"must have either just one port or as many as DP × CP degree "
        #     f"({len(port)} given, but {dp_size} × {cp_size} = {dp_cp_size})"
        # )

        # self.init_serving()
        # if dp_cp_rank >= 0:
        #     # Automatically extend port list from given base port.
        #     if len(port) != dp_cp_size:
        #         port = [port[0] + i for i in range(dp_cp_size)]

        #     port = port[dp_cp_rank]
        #     self.start_server(address, port)
        #     logger.info(f"Serving model at {address}:{port}.")
        #     self.serve_forever(poll_interval)
        # else:
        #     while True:
        #         self.before_process_request()
        #         input_dict = dict(input=torch.empty(2, 2, 2).to(self.device))
        #         outputs = self.forward_and_gather(input_dict)
        #         assert outputs is None
        #         self.after_process_request()

        assert isinstance(port, int) or len(port) == 1
        if isinstance(port, list):
            port = port[0]

        if (
            self.parallel_dims.dp_replicate_enabled
            or self.parallel_dims.dp_shard_enabled
        ):
            dp_rank = self.get_group_rank("dp")
        else:
            dp_rank = 0
        self.init_serving()
        if dp_rank >= 0:
            self.start_server(address, port)
            logger.info(f"Serving model at {address}:{port}.")
            self.serve_forever(poll_interval)
        else:
            while True:
                self.before_process_request()
                # input_dict = torch.gather TODO???
                # TODO remove line below
                input_dict = dict(input=torch.empty(2, 2, 2).to(self.device))
                outputs = self.forward_and_gather(input_dict)
                assert outputs is None
                self.after_process_request()

    def wait_for_inputs(self):
        # input_dict = torch.gather TODO???
        return input_dict

    def before_process_request(self):
        print("waiting for before barrier")
        torch.distributed.barrier()
        print("passed before barrier")
        self.step += 1
        self.gc_handler.run(self.step)

    def process_request(self, request, client_address):
        self.before_process_request()
        super().process_request(request, client_address)
        self.after_process_request()

    def after_process_request(self):
        print("waiting for after barrier")
        torch.distributed.barrier()
        print("passed after barrier")
        if self.step == 1:
            dist_utils.set_pg_timeouts(
                timeout=timedelta(
                    seconds=self.job_config.comm.train_timeout_seconds
                ),
                world_mesh=self.world_mesh,
            )

    def init_serving(self):
        self.step = 0

        self.activations_handling_ctx = get_act_offloading_ctx_manager(
            self.model_parts[0], enable_activation_offloading=False
        )

        self.eval_context = dist_utils.get_train_context(
            enable_loss_parallel=False,
            enable_compiled_autograd=False,
        )

    def serve_forever(self, poll_interval=0.5):
        super().serve_forever(poll_interval)

        if torch.distributed.get_rank() == 0:
            logger.info("Sleeping 2 seconds for other ranks to complete")
            time.sleep(2)

        self.metrics_processor.close()
        logger.info("Serving completed")

    def get_group(self, name):
        group = self.world_mesh[name].get_group()
        return group

    def get_group_rank(self, name):
        group = self.get_group(name)
        group_rank = torch.distributed.get_rank(group=group)
        return group_rank

    def get_group_size(self, name):
        group = self.get_group(name)
        group_size = torch.distributed.get_world_size(group=group)
        return group_size

    def in_group(self, name):
        group_rank = self.get_group_rank(name)
        # rank is -1 if process is not part of group
        return group_rank >= 0

    @torch.no_grad()
    def forward_and_gather(
            self,
            input_dict: dict[str, torch.Tensor],
            first_seq_elem_indices: torch.Tensor,
    ):
        outputs = self.forward(input_dict)
        if self.parallel_dims.pp_enabled and not self.pp_has_last_stage:
            # Only in the PP case should we ever have `None` outputs:
            # for all stages that are not (a replica of) the last one.
            assert outputs is None
            first_seq_elem_indices = None
        else:
            assert outputs is not None
            if isinstance(outputs, dict):
                outputs = outputs["tokens_list"]
            if isinstance(outputs, list):
                outputs = outputs[0]
            if isinstance(outputs, torch.Tensor):
                outputs = outputs.detach().cpu()

            # Gather in the CP dimension. CP ranks = 0 should also be
            # the only parts in the DP group.
            if self.parallel_dims.cp_enabled:
                cp_group = self.get_group("cp")
                cp_group_rank = self.get_group_rank("cp")

                gather_list = None
                if cp_group_rank == 0:
                    gather_list = [
                        torch.empty_like(outputs)
                        for _ in self.get_group_size("cp")
                    ]

                torch.distributed.gather(
                    outputs,
                    gather_list=gather_list,
                    group=cp_group,
                    group_dst=0,
                )
                if cp_group_rank == 0:
                    outputs = torch.cat([t for t in gather_list], dim=1)
                else:
                    outputs = None
                del gather_list

            # Gather in the DP dimension. All the CP ranks = 0 should
            # have entire sequences at this point.
            if self.parallel_dims.dp_enabled:
                dp_group = self.get_group("dp")
                dp_group_rank = self.get_group_rank("dp")

                gather_list = None
                if dp_group_rank == 0:
                    gather_list = [
                        torch.empty_like(outputs)
                        for _ in self.get_group_size("dp")
                    ]

                torch.distributed.gather(
                    outputs,
                    gather_list=gather_list,
                    group=dp_group,
                    group_dst=0,
                )
                if dp_group_rank == 0:
                    outputs = torch.cat([t for t in gather_list], dim=0)
                else:
                    outputs = None
                del gather_list

                # Gather first seq elem indices to remove padding from
                # gathered tensors.
                gather_list = None
                if dp_group_rank == 0:
                    gather_list = [
                        torch.empty_like(first_seq_elem_indices)
                        for _ in self.get_group_size("dp")
                    ]

                torch.distributed.gather(
                    first_seq_elem_indices,
                    gather_list=gather_list,
                    group=dp_group,
                    group_dst=0,
                )
                if dp_group_rank == 0:
                    first_seq_elem_indices = torch.cat([t for t in gather_list], dim=0)
                else:
                    first_seq_elem_indices = None
                del gather_list
        return outputs, first_seq_elem_indices

    @torch.inference_mode()
    def forward(self, input_dict: dict[str, torch.Tensor]):
        model_parts = self.model_parts
        world_mesh = self.world_mesh
        parallel_dims = self.parallel_dims

        # apply context parallelism if cp is enabled
        # ensure CP handles the separate freqs_cis buffer for each pp stage
        inputs = input_dict["input"]
        optional_context_parallel_ctx = (
            dist_utils.create_context_parallel_ctx(
                cp_mesh=world_mesh["cp"],
                cp_buffers=[inputs] + [m.freqs_cis for m in model_parts],
                cp_seq_dims=[1] + [0 for _ in model_parts],
                cp_no_restore_buffers={inputs},
                cp_rotate_method=self.job_config.parallelism.context_parallel_rotate_method,
            )
            if parallel_dims.cp_enabled
            else None
        )

        inputs = dict(tokens_list=inputs, start_pos=input_dict.get("start_pos", 0))
        if parallel_dims.pp_enabled:
            # Pipeline Parallel forward / backward inside step() call
            pred = None
            with self.eval_context(optional_context_parallel_ctx, self.activations_handling_ctx):
                if self.pp_has_first_stage:
                    self.pp_schedule.step(inputs)
                elif self.pp_has_last_stage:
                    pred = self.pp_schedule.step()
                    # TODO: PP+FSDP unexpectedly puts the loss back to the CPU;
                    #       check if this is a problem for the outputs, too.
                    #       Right now we just move for safety.
                    if pred.device != self.device:
                        warnings.warn("Wrong device after PP+FSDP step; moving tensor...")
                    pred = pred.to(self.device)
                else:
                    self.pp_schedule.step()
        else:
            # Non-PP forward / backward
            with self.eval_context(optional_context_parallel_ctx):
                assert len(model_parts) == 1
                pred = model_parts[0](inputs)

        return pred

    def init_model(self, job_config: JobConfig):
        self.job_config = job_config

        if job_config.experimental.custom_import:
            importlib.import_module(job_config.experimental.custom_import)

        if job_config.job.print_args:
            logger.info(f"Running with args: {job_config.to_dict()}")

        # take control of garbage collection to avoid stragglers
        self.gc_handler = utils.GarbageCollection(gc_freq=job_config.training.gc_freq)

        device_module, device_type = utils.device_module, utils.device_type
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        # Device has to be set before creating TorchFT manager.
        device_module.set_device(self.device)

        # init distributed
        world_size = int(os.environ["WORLD_SIZE"])
        parallelism_config = job_config.parallelism
        self.parallel_dims = parallel_dims = ParallelDims(
            dp_shard=parallelism_config.data_parallel_shard_degree,
            dp_replicate=parallelism_config.data_parallel_replicate_degree,
            cp=parallelism_config.context_parallel_degree,
            tp=parallelism_config.tensor_parallel_degree,
            pp=parallelism_config.pipeline_parallel_degree,
            ep=parallelism_config.expert_parallel_degree,
            ep_mode=parallelism_config.expert_parallel_mode,
            world_size=world_size,
            enable_loss_parallel=not parallelism_config.disable_loss_parallel,
        )
        dist_utils.init_distributed(job_config, self.device)

        # build meshes
        self.world_mesh = world_mesh = parallel_dims.build_mesh(device_type=device_type)

        self.train_spec = train_spec_module.get_train_spec(job_config.model.name)

        self.tokenizer = tokenizer = (
            self.train_spec.build_tokenizer_fn(job_config)
            if self.train_spec.build_tokenizer_fn is not None
            else None
        )

        model_cls = self.train_spec.cls
        model_args = self.train_spec.config[job_config.model.flavor]
        # set the model args from training job configs
        model_args.update_from_config(job_config, tokenizer)

        logger.info(
            f"Building {self.train_spec.name} {job_config.model.flavor} with {model_args}"
        )

        with torch.device("meta"):
            model = model_cls.from_model_args(model_args)

        logger.info(f"model: {model}")

        model_converters = build_model_converters(job_config, parallel_dims)
        model_converters.convert(model)

        # metrics logging
        build_metrics_processor_fn = (
            build_metrics_processor
            if self.train_spec.build_metrics_processor_fn is None
            else self.train_spec.build_metrics_processor_fn
        )
        self.metrics_processor = build_metrics_processor_fn(job_config, parallel_dims)
        color = self.metrics_processor.color

        # calculate model size and flops per token
        (
            model_active_param_count,
            model_param_count,
            self.metrics_processor.num_flops_per_token,
        ) = model_args.get_nparams_and_flops(model, job_config.training.seq_len)

        logger.info(
            f"{color.blue}Model {self.train_spec.name} {job_config.model.flavor} "
            f"{color.red}size: {model_param_count:,} total parameters, "
            f"{model_active_param_count:,} active parameters{color.reset}"
        )

        # move sharded model to CPU/GPU and initialize weights via DTensor
        if job_config.checkpoint.create_seed_checkpoint:
            init_device = "cpu"
            buffer_device = None
        elif job_config.training.enable_cpu_offload:
            init_device = "cpu"
            buffer_device = device_type
        else:
            init_device = device_type
            buffer_device = None

        # apply parallelisms and initialization
        if parallel_dims.pp_enabled:
            if not self.train_spec.pipelining_fn:
                raise RuntimeError(
                    f"Pipeline Parallel is enabled but {self.train_spec.name} "
                    f"does not support pipelining"
                )

            with force_pipeline_schedule(schedules._ScheduleForwardOnly):
                # apply both PT-D Pipeline Parallel and SPMD-style PT-D techniques
                (
                    self.pp_schedule,
                    self.model_parts,
                    self.pp_has_first_stage,
                    self.pp_has_last_stage,
                ) = self.train_spec.pipelining_fn(
                    model,
                    world_mesh,
                    parallel_dims,
                    job_config,
                    self.device,
                    model_args,
                    self.train_spec.parallelize_fn,
                    self.loss_fn,
                )
            # when PP is enabled, `model` obj is no longer used after this point,
            # model_parts is used instead
            del model

            for m in self.model_parts:
                m.to_empty(device=init_device)
                with torch.no_grad():
                    m.init_weights(buffer_device=buffer_device)
                m.eval()
        else:
            # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
            model = self.train_spec.parallelize_fn(
                model, world_mesh, parallel_dims, job_config
            )

            model.to_empty(device=init_device)
            with torch.no_grad():
                model.init_weights(buffer_device=buffer_device)
            model.eval()

            self.model_parts = [model]

        # initialize device memory monitor and get peak flops for MFU calculation
        device_memory_monitor = self.metrics_processor.device_memory_monitor
        gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
        logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")
        device_mem_stats = device_memory_monitor.get_peak_stats()
        logger.info(
            f"{device_type.upper()} memory usage for model: "
            f"{device_mem_stats.max_reserved_gib:.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)"
        )

    def load_model(self, checkpoint_folder: str):
        states = {STATE_MODEL_KEY: ModelWrapper(self.model_parts)}
        dcp.load(states, checkpoint_id=checkpoint_folder)

    def close(self):
        atexit.unregister(self.server_close)


def main(args_list: list[str] | None = None):
    args = parse_args(args_list)
    if all([args.dump_folder, args.checkpoint_folder, args.job_config_file]):
        warnings.warn(
            (
                "Both `--checkpoint_folder` and `--job_config_file` have "
                "been given; `--dump_folder` will be completely ignored."
            ),
        )
    elif not args.dump_folder and (not args.checkpoint_folder or not args.job_config_file):
        raise ValueError(
            (
                "Both `--checkpoint_folder` and `--job_config_file` need to "
                "be given if `--dump_folder` is not."
            ),
        )

    if args.job_config_file:
        job_config_file = args.job_config_file
    else:
        job_config_file_re = re.compile(r"job_config_[0-9]{8}-[0-9]{4}\.json")
        job_config_files = sorted(filter(
            lambda p: job_config_file_re.match(p) is not None,
            os.listdir(args.dump_folder),
        ))

        job_config_file = os.path.join(args.dump_folder, job_config_files[-1])
        if len(job_config_files) > 1:
            warnings.warn(
                f"Found multiple job config files in dump folder; using latest one "
                f"(`{job_config_file}`)... "
                f"Use `--job_config_file` to specify a specific one."
            )
    with open(job_config_file, "r") as f:
        job_config_dict = json.load(f)
    job_config = JobConfig()
    for (k, v) in job_config_dict.items():
        class_type = type(k.title(), (), v)
        setattr(job_config, k, class_type())

    if args.checkpoint_folder:
        checkpoint_folder = args.checkpoint_folder
    else:
        checkpoint_folder_re = re.compile(r"step-[0-9]+")
        checkpoint_parent_folder = os.path.join(args.dump_folder, job_config.checkpoint.folder)
        checkpoint_folders = sorted(filter(
            lambda p: checkpoint_folder_re.match(p) is not None,
            os.listdir(checkpoint_parent_folder),
        ))

        checkpoint_folder = os.path.join(checkpoint_parent_folder, checkpoint_folders[-1])
        if len(checkpoint_folders) > 1:
            warnings.warn(
                f"Found multiple checkpoint folders in dump folder; using latest one "
                f"(`{checkpoint_folder}`)... "
                f"Use `--checkpoint_folder` to specify a specific one."
            )

    # Patch job config
    # We don't want/need any pure data parallelism.
    job_config.parallelism.data_parallel_replicate_degree = 1
    # We don't care about activation checkpointing.
    job_config.activation_checkpoint.mode = "none"
    # We need FlexAttention for KV caching.
    job_config.model.use_flex_attn = True
    # Disable compilation if not available.
    device_type = utils.device_type
    if not has_triton():
        if (
            device_type == "cuda"
            and (device_props := torch.cuda.get_device_properties(device_type)).major < 7
        ):
            logging.warning(
                f"Found {device_props.name} which is too old to be supported by the triton GPU "
                f"compiler, which is used as the backend. Triton only supports devices of CUDA "
                f"Capability >= 7.0, but your device is of CUDA capability "
                f"{device_props.major}.{device_props.minor}. "
                f"Disabling compilation..."
            )
        elif is_gpu(device_type):
            logging.warning(
                "Cannot find a working triton installation. Either the package is not installed or "
                "it is too old. More information on installing Triton can be found at "
                "https://github.com/openai/triton"
            )

        job_config.training.compile = False

        # Patch FlexAttention to remove hardcoded `torch.compile`d
        # functions.
        from torchtitan.models.attention import FlexAttention

        from torch.nn.attention.flex_attention import (
            create_block_mask,
            flex_attention,
        )

        FlexAttention.compiled_create_block_mask = create_block_mask
        FlexAttention.flex_attn = flex_attention

    server = None
    try:
        server = TorchTitanServer(job_config, checkpoint_folder)
        server.serve(args.server_address, args.server_port, logits_only=not args.do_sampling)
    finally:
        if server:
            server.close()

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            logger.info("Process group destroyed.")


if __name__ == "__main__":
    init_logger()
    log_level = logging.DEBUG
    logger.setLevel(log_level)
    for handler in logger.handlers:
        handler.setLevel(log_level)
    main()
