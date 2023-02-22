"""Run a chatbot with FlexGen and OPT models."""
from typing import Tuple

import fire as fire
from transformers import AutoTokenizer
from flexgen.flex_opt import (
    Policy, OptLM, TorchDevice, TorchDisk, TorchMixedDevice, CompressionConfig, Env, get_opt_config
)


def main(
    model: str = "facebook/opt-6.7b",
    path: str = "~/opt_weights",
    offload_dir: str = "~/flexgen_offload_dir",
    percent: Tuple[int, int, int, int, int, int] = (100, 0, 100, 0, 100, 0),
    compress_weight: bool = False,
    compress_cache: bool = False,
):
    """The main entry point.

    Args:
        model: The model name.
        path: The path to the model weights. If there are no cached weights,
            FlexGen will automatically download them from HuggingFace.
        offload_dir: The directory to offload tensors.
        percent: Six numbers. They are:

            + the percentage of weight on GPU
            + the percentage of weight on CPU
            + the percentage of attention cache on GPU
            + the percentage of attention cache on CPU
            + the percentage of activations on GPU
            + the percentage of activations on CPU

        compress_weight: Whether to compress weight.
        compress_cache: Whether to compress cache.
    """
    # Initialize environment
    gpu = TorchDevice("cuda:0")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(offload_dir)
    env = Env(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))

    # Offloading policy
    policy = Policy(1, 1,
                    percent[0], percent[1],
                    percent[2], percent[3],
                    percent[4], percent[5],
                    overlap=True, sep_layer=True, pin_weight=True,
                    cpu_cache_compute=False, attn_sparsity=1.0,
                    compress_weight=compress_weight,
                    comp_weight_config=CompressionConfig(
                        num_bits=4, group_size=64,
                        group_dim=0, symmetric=False),
                    compress_cache=compress_cache,
                    comp_cache_config=CompressionConfig(
                        num_bits=4, group_size=64,
                        group_dim=2, symmetric=False))

    # Model
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", padding_side="left")
    tokenizer.add_bos_token = False
    stop = tokenizer("\n").input_ids[0]

    print("Initialize...")
    opt_config = get_opt_config(model)
    model = OptLM(opt_config, env, path, policy)
    model.init_all_weights()

    context = (
        "A chat between a curious human and a knowledgeable artificial intelligence assistant.\n"
        "Human: Hello! What can you do?\n"
        "Assistant: As an AI assistant, I can answer questions and chat with you.\n"
        "Human: What is the name of the tallest mountain in the world?\n"
        "Assistant: Everest.\n"
    )

    # Chat
    print(context, end="")
    while True:
        inp = input("Human: ")
        if not inp:
            print("exit...")
            break

        context += "Human: " + inp + "\n"
        inputs = tokenizer([context])
        output_ids = model.generate(
            inputs.input_ids,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=96,
            stop=stop)
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        try:
            index = outputs.index("\n", len(context))
        except ValueError:
            outputs += "\n"
            index = outputs.index("\n", len(context))
        
        outputs = outputs[:index + 1]
        print(outputs[len(context):], end="")
        context = outputs

    # TODO: optimize the performance by reducing redundant computation.

    # Shutdown
    model.delete_all_weights()
    disk.close_copy_threads()


if __name__ == "__main__":
    fire.Fire(main)
