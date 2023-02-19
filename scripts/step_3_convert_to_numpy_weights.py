# Copied from
# https://github.com/alpa-projects/alpa/blob/main/examples/llm_serving/scripts/step_3_convert_to_numpy_weights.py

"""Convert Metaseq's OPT model weights into Alpa numpy weights."""
import time

import argparse
import os

import numpy as np
from scripts.utils import torch_load_cpu


def save_numpy(weight_dict, to_folder):
    os.makedirs(to_folder, exist_ok=True)
    for tensor_name, tensor in weight_dict.items():
        print(f"- Writing tensor {tensor_name} with shape {tensor.shape}")
        t = tensor.cpu().detach().numpy()
        with open(to_folder + "/" + tensor_name, "wb") as g:
            np.save(g, t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, default="/home/ubuntu/consolidated")
    parser.add_argument("--output-folder", type=str, default="/home/ubuntu/opt-175b-np")
    args = parser.parse_args()
    start_time = time.time()
    print("- Reading the weight into memory")
    state = torch_load_cpu(args.ckpt_path)
    print(f"Done with reading: {time.time() - start_time} seconds")
    save_numpy(state["model"], args.output_folder)
