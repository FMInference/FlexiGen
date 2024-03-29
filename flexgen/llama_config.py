"""
The Llama model configurations and weight downloading utilities.

adopted from opt_config.py
"""

import dataclasses
import glob
import os
import numpy as np
from tqdm import tqdm


@dataclasses.dataclass(frozen=True)
class LlamaConfig:
    name: str = "Llama-2-7b-hf"
    hf_token: str = ''
    hidden_act: str = "silu"
    input_dim: int = 4096
    initializer_range: float = 0.02
    intermediate_size: int = 11008
    max_position_embeddings: int = 4096
    n_head: int = 32
    num_hidden_layers: int = 32
    num_key_value_heads: int = 32
    rms_norm_eps: float = 1e-05
    dtype: type = np.float16
    pad_token_id: int = 2
    vocab_size: int = 32000

    def model_bytes(self):
        h = self.input_dim
        intermediate = self.intermediate_size
        n_head = self.n_head
        head_dim = h // n_head
        return 2 * (self.vocab_size * h +
        self.num_hidden_layers * (
        # self-attention
        3 * h * h + h * h + head_dim // 2 +
        # mlp
        3 * h * intermediate +
        # layer norm
        2 * h) +
        # head
        h + self.vocab_size * h)

    def cache_bytes(self, batch_size, seq_len):
        return 2 * batch_size * seq_len * self.num_hidden_layers * self.input_dim * 2

    def hidden_bytes(self, batch_size, seq_len):
        return batch_size * seq_len * self.input_dim * 2


def get_llama_config(name, **kwargs):
    if "/" in name:
        name = name.split("/")[1]

    if "-chat" in name:
        arch_name = name.replace("-chat", "")
    else:
        arch_name = name

    if arch_name == "Llama-2-7b-hf":
        config = LlamaConfig(name=name, hf_token=kwargs.get('hf_token'),
                             input_dim=4096, intermediate_size=11008, n_head=32,
                             num_hidden_layers=32, num_key_value_heads=32
                             )
    elif arch_name == "Llama-2-13b-hf":
        config = LlamaConfig(name=name, hf_token=kwargs.get('hf_token'),
                             input_dim=5120, intermediate_size=13824, n_head=40,
                             num_hidden_layers=40, num_key_value_heads=40
                             )
    elif arch_name == "Llama-2-70b-hf":
        config = LlamaConfig(name=name, hf_token=kwargs.get('hf_token'),
                             input_dim=8192, intermediate_size=28672, n_head=64,
                             num_hidden_layers=80, num_key_value_heads=8
                             )
    else:
        raise ValueError(f"Invalid model name: {name}")

    return dataclasses.replace(config, **kwargs)


def download_llama_weights(model_name, path, hf_token):
    from huggingface_hub import snapshot_download
    import torch

    print(f"Load the pre-trained pytorch weights of {model_name} from huggingface. "
          f"The downloading and cpu loading can take dozens of minutes. "
          f"If it seems to get stuck, you can monitor the progress by "
          f"checking the memory usage of this process.")

    hf_model_name = "meta-llama/" + model_name

    folder = snapshot_download(hf_model_name, allow_patterns="*.bin", token=hf_token)
    bin_files = glob.glob(os.path.join(folder, "*.bin"))

    if "/" in model_name:
        model_name = model_name.split("/")[1]
    path = os.path.join(path, f"{model_name}-np")
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(path, exist_ok=True)

    for bin_file in tqdm(bin_files, desc="Convert format"):
        state = torch.load(bin_file, map_location='cuda:0')
        for name, param in tqdm(state.items(), leave=False):
            name = name.replace("model.", "")
            param_path = os.path.join(path, name)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().numpy())
