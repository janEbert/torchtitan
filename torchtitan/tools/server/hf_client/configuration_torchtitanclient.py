from transformers import PretrainedConfig

from torchtitan.tools.server.serve_model import DEFAULT_PORT


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
