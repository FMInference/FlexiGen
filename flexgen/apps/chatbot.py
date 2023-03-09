# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.
"""

python3 apps/chatbot.py --model facebook/opt-1.3b --platform cpu
python3 apps/chatbot.py --model facebook/opt-2.7b --platform mps:0
python3 apps/chatbot.py --model facebook/opt-1.3b --platform mps:0 --dpl-apikey YOUR_API_KEY --dpl-lang de|ja|...
python3 apps/chatbot.py --model facebook/opt-66b --percent 100 0 100 0 100 0 --compress-weight --platform mps:0

--gen-len 96

DeepL API Request Parameters
https://www.deepl.com/ja/docs-api/translate-text/translate-text/

"""

import argparse
import time

import numpy as np

from flexgen.flex_opt import (Policy, OptLM, ExecutionEnv, CompressionConfig, str2bool)
from flexgen.pytorch_backend import (TorchDevice, TorchDisk, TorchMixedDevice)
from flexgen.opt_config import get_opt_config

from transformers import AutoTokenizer
import torch
import deepl


def run_chat(args):
        
    # Initialize environment

    if args.platform == "cpu":
        gpu = TorchDevice("cpu")
    else:
        gpu = TorchDevice(args.platform)
    cpu = TorchDevice("cpu")
    disk = TorchDisk(args.offload_dir,platform=args.platform)

    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]), platform=args.platform)

    # Offloading policy
    policy = Policy(1, 1,
                    args.percent[0], args.percent[1],
                    args.percent[2], args.percent[3],
                    args.percent[4], args.percent[5], args.platform == "cpu",
                    args.overlap, args.sep_layer, args.pin_weight,
                    args.cpu_cache_compute, args.attn_sparsity,
                    args.compress_weight,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=0, symmetric=False),
                    args.compress_cache,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=2, symmetric=False))
    assert not (args.compress_cache and args.attn_sparsity < 1.0), "Not implemented"

    # DeepL setting
    if not args.dpl_apikey == "":
        dpl_translator = deepl.Translator(args.dpl_apikey)

    # Model
    print("Initialize...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    tokenizer.add_bos_token = False
    stop = tokenizer("\n").input_ids[0]

    opt_config = get_opt_config(args.model, dtype=np.float32 if args.platform == "cpu" else np.float16)
    model_nm = OptLM(opt_config, env, args.path, policy)

    context = (
        "A chat between a curious human and a knowledgeable artificial intelligence assistant.\n"
        "Human: Hello! What can you do?\n"
        "Assistant: As an AI assistant, I can answer questions and chat with you.\n"
        "Human: What is the name of the tallest mountain in the world?\n"
        "Assistant: Everest.\n"
    )

    # Chat
    # print("\n================================== Start\n")
    print(context, end="")
    while True:
        inp = input("Human: ")
        if not inp:
            print("exit...")
            # print("\n================================== End\n")
            break

        start_time = time.time()

        # DeepL Translate
        if not args.dpl_apikey == "":
            trns_inp = dpl_translator.translate_text(inp, target_lang="EN-US")
            inp = trns_inp.text

        context += "Human: " + inp + "\n"
        inputs = tokenizer([context])
        output_ids = model_nm.generate(inputs.input_ids,
                                        max_new_tokens=args.gen_len,
                                        do_sample=True,
                                        temperature=0.7,
                                        stop=stop,
                                        debug_mode=args.debug_mode,
                                        cut_gen_len=args.cut_gen_len,
                                        verbose=args.verbose)
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        try:
            index = outputs.index("\n", len(context))
        except ValueError:
            outputs += "\n"
            index = outputs.index("\n", len(context))

        # outputs: All text
        # context: Last text
        # index: len(context) + len(reply from model)
        
        # DeepL Translate
        outputs = outputs[:index + 1]
        if (not args.dpl_apikey == "") and (not args.dpl_lang == ""):
            out = outputs[len(context):].strip("\n").replace("Assistant:", "")
            trns_out = dpl_translator.translate_text(out, target_lang=args.dpl_lang)
            output = trns_out.text
            print("Assistant: ", end="")

        else:
            output = outputs[len(context):].strip("\n")

        print(output, end="")
        print(f" [{time.time() - start_time:.2f}s]")
        context = outputs

    # TODO: optimize the performance by reusing context cache and reducing redundant computation.

    # Shutdown
    env.close_copy_threads()



def add_parser_arguments(parser):
    parser.add_argument("--model", type=str, default="facebook/opt-6.7b",
        help="The model name.")
    parser.add_argument("--path", type=str, default="~/opt_weights",
        help="The path to the model weights. If there are no cached weights, "
             "FlexGen will automatically download them from HuggingFace.")
    parser.add_argument("--offload-dir", type=str, default="~/flexgen_offload_dir",
        help="The directory to offload tensors. ")
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--gen-len", type=int, default=64)
    parser.add_argument("--cut-gen-len", type=int,
        help="Cut generation length for fast debugging.")
    parser.add_argument("--num-gpu-batches", type=int, default=1)
    parser.add_argument("--debug-mode", type=str,
        choices=["fewer_batch", "breakdown"])
    parser.add_argument("--gpu-batch-size", type=int, default=4)
    parser.add_argument("--percent", nargs="+", type=int,
        default=[100, 0, 100, 0, 100, 0],
        help="Six numbers. They are "
         "the percentage of weight on GPU, "
         "the percentage of weight on CPU, "
         "the percentage of attention cache on GPU, "
         "the percentage of attention cache on CPU, "
         "the percentage of activations on GPU, "
         "the percentage of activations on CPU")
    parser.add_argument("--sep-layer", type=str2bool, nargs='?',
        const=True, default=True)
    parser.add_argument("--pin-weight", type=str2bool, nargs="?",
        const=True, default=True)
    parser.add_argument("--cpu-cache-compute", action="store_true")
    parser.add_argument("--attn-sparsity", type=float, default=1.0)
    parser.add_argument("--compress-weight", action="store_true",
        help="Whether to compress weight.")
    parser.add_argument("--compress-cache", action="store_true",
        help="Whether to compress cache.")


    parser.add_argument("--log-file", type=str, default="auto")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--verbose", type=int, default=2)

    parser.add_argument("--overlap", type=str2bool, nargs='?',
        const=True, default=True)

    parser.add_argument("--platform", type=str, default="cuda:0", help="use the number to specify device, the platform can also be cpu or mps")

    parser.add_argument("--dpl-apikey", type=str, default="", help="DeepL ApiKey")
    parser.add_argument("--dpl-lang", type=str, default="", help="DeepL translates Assistant reply")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    args = parser.parse_args()

    assert len(args.percent) == 6

    if "cuda" in args.platform:
        if not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                args.platform = "mps:0"
            else:
                args.platform = "cpu"
            print("CUDA devices not available, {} is used instead".format(args.platform))

    if "mps" in args.platform:
        if not torch.backends.mps.is_available():
            args.platform = "cpu"
            print("MPS devices not available, CPU is used instead")

    if "cuda" not in args.platform:
        # not clear how to enable overlap on MPS platform yet
        args.overlap = False
        args.pin_weight = False

    if args.platform == "cpu":
        args.percent = [0, 100, 0, 100, 0, 100]    

    run_chat(args)