# The source code in this file is partially adapted from
# https://github.com/HazyResearch/fm_data_tasks/blob/main/fm_data_tasks/utils/prompt_utils.py
# which is under Apache License Version 2.0.

"""Run inference."""
import argparse
from tqdm import tqdm
import json
import math
import logging
from pathlib import Path
import time
import numpy as np
from transformers import AutoTokenizer, AutoConfig
import flexgen.apps.data_wrangle.utils.data_utils as data_utils
import flexgen.apps.data_wrangle.utils.prompt_utils as prompt_utils
from flexgen.apps.data_wrangle.utils import constants
from flexgen.apps.data_wrangle.utils.utils import compute_metrics, setup_logger
from flexgen.flex_opt import (Policy, OptLM, ExecutionEnv, CompressionConfig, str2bool)


logger = logging.getLogger(__name__)


def add_flexgen_args(parser):
    parser.add_argument("--pad-to-seq-len", type=int)
    parser.add_argument("--model", type=str, default="facebook/opt-1.3b",
        help="The model name.")
    parser.add_argument("--path", type=str, default="~/opt_weights",
        help="The path to the model weights. If there are no cached weights, "
             "FlexGen will automatically download them from HuggingFace.")
    parser.add_argument("--run-path", type=str, default="runs")
    parser.add_argument("--offload-dir", type=str, default="~/flexgen_offload_dir",
        help="The directory to offload tensors. ")
    parser.add_argument("--gpu-batch-size", type=int, default=16)
    parser.add_argument("--num-gpu-batches", type=int, default=1)
    parser.add_argument("--percent", nargs="+", type=int,
        default=[100, 0, 100, 0, 100, 0],
        help="Six numbers. They are "
         "the percentage of weight on GPU, "
         "the percentage of weight on CPU, "
         "the percentage of attention cache on GPU, "
         "the percentage of attention cache on CPU, "
         "the percentage of activations on GPU, "
         "the percentage of activations on CPU")
    parser.add_argument("--pin-weight", type=str2bool, nargs="?",
        const=True, default=True)
    parser.add_argument("--cpu-cache-compute", action="store_true")
    parser.add_argument("--compress-weight", action="store_true",
        help="Whether to compress weight.")
    parser.add_argument("--compress-cache", action="store_true",
        help="Whether to compress cache.")


def parse_args() -> argparse.Namespace:
    """Generate args."""
    parser = argparse.ArgumentParser(description="Simple calculator")
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Which data directory to run.",
        required=True,
    )
    parser.add_argument(
        "--output_dir", type=str, help="Output directory.", default="outputs"
    )
    parser.add_argument(
        "--cache_name",
        type=str,
        help="Manifest cache type.",
        default="sqlite",
        choices=["redis", "sqlite", "noop"],
    )
    parser.add_argument(
        "--cache_connection",
        type=str,
        help="Manifest cache connection string.",
        default="fm_data_tasks.sqlite",
    )
    parser.add_argument(
        "--client_name",
        type=str,
        help="Manifest client type.",
        default="openai",
        choices=["openai", "opt", "huggingface"],
    )
    parser.add_argument(
        "--client_connection",
        type=str,
        help="Manifest client connection string.",
        default=None,
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        help="Tag for run saving.",
        default="default",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite sqlite cache of input/output results.",
    )
    parser.add_argument("--k", type=int, help="Number examples in prompt", default=1)
    parser.add_argument(
        "--sample_method",
        type=str,
        help="Example generation method",
        default="random",
        choices=["random", "manual", "validation_clusters"],
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--class_balanced",
        help="Class balance training data. Good for classification tasks \
             with random prompts.",
        action="store_true",
    )
    parser.add_argument(
        "--sep_tok",
        type=str,
        help="Separate for attr: val pairs in row. Default is '.'.",
        default=".",
    )
    parser.add_argument(
        "--nan_tok",
        type=str,
        help="Token to represent nan entries. Default is 'nan'.",
        default="nan",
    )
    parser.add_argument(
        "--num_run",
        type=int,
        help="Number examples to run through model.",
        default=-1,
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        help="Number trials to run. Results will be averaged with variance reported.",
        default=1,
    )
    parser.add_argument(
        "--num_print",
        type=int,
        help="Number example prompts to print.",
        default=10,
    )
    parser.add_argument(
        "--add_task_instruction",
        help="Add task instruction to the prompt before examples.",
        action="store_true",
    )
    parser.add_argument("--task_instruction_idx", type=int, default=0)
    parser.add_argument("--do_test", help="Run on test file.", action="store_true")
    parser.add_argument(
        "--dry_run", help="Dry run. Do not actually ping model.", action="store_true"
    )

    parser.add_argument(
        "--stop_token", help="Token to stop on for a given generated response", default="\n"
    )

    # Model args
    parser.add_argument("--temperature", type=float, help="Temperature.", default=0.0)
    parser.add_argument(
        "--max_tokens", type=int, help="Max tokens to generate.", default=3
    )
    
    parser.add_argument(
        "--batch_run", help="Use FlexGen batch inference.", action="store_true"
    )
    
    add_flexgen_args(parser)
    
    args = parser.parse_args()
    return args


def get_tokenizer(name):
    if name == 'facebook/opt-175b':
        tokenizer = AutoTokenizer.from_pretrained('facebook/opt-30b', padding_side="left")
    else:
        tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
    tokenizer.add_bos_token = False
    if 'galactica' in name:
        config = AutoConfig.from_pretrained(name)
        tokenizer.pad_token = config.pad_token_id
        tokenizer.eos_token = config.eos_token_id
    return tokenizer


def single_query_test(args, task_instruction, test_data, task, pd_data_files, test_file):
    # Initialize environment
    tokenizer = get_tokenizer(args.model)
    env = ExecutionEnv.create(args.offload_dir)

    # Offloading policy
    policy = Policy(1, 1,
                    args.percent[0], args.percent[1],
                    args.percent[2], args.percent[3],
                    args.percent[4], args.percent[5],
                    overlap=True, sep_layer=True, pin_weight=args.pin_weight,
                    cpu_cache_compute=args.cpu_cache_compute, attn_sparsity=1.0,
                    compress_weight=args.compress_weight,
                    comp_weight_config=CompressionConfig(
                        num_bits=4, group_size=64,
                        group_dim=0, symmetric=False),
                    compress_cache=args.compress_cache,
                    comp_cache_config=CompressionConfig(
                        num_bits=4, group_size=64,
                        group_dim=2, symmetric=False))

    logger.info(f"Init weights begin.")
    tic = time.time()
    model = OptLM(args.model, env, args.path, policy)
    logger.info(f"Init weights end. Elapsed: {time.time() - tic:.2f} s")

    if args.add_task_instruction:
        prompt = lambda x: f"{task_instruction} {x}"
    else:
        prompt = lambda x: f"{x}"
    trial_metrics = {"prec": [], "rec": [], "f1": [], "acc": []}

    saved_prefix = None
    
    tic = time.time()
    for trial_num in range(args.num_trials):
        np.random.seed(args.seed + trial_num)
        queries = []
        for _, row in test_data.iterrows():
            serialized_r = row["text"]
            if args.sample_method == "manual":
                prefix_exs = prompt_utils.get_manual_prompt(args.data_dir, row)
            elif args.sample_method == "validation_clusters":
                if saved_prefix is None:
                    logger.info("Generating validation cluster prompt.")
                    saved_prefix = prompt_utils.get_validation_prompt(
                        args.validation_path,
                        num_examples=args.k,
                        task=task,
                    )
                prefix_exs = saved_prefix
            else:
                if saved_prefix is None:
                    saved_prefix = prompt_utils.get_random_prompt(
                        pd_data_files["train"], num_examples=args.k
                    )
                prefix_exs = saved_prefix
            queries.append((prefix_exs + "\n" + serialized_r).strip())

        gt = test_data["label_str"]
        preds = []
        idx = 0
        for _ in range(args.num_print):
            logger.info(prompt(queries[idx]))
            tic = time.time()
            input_ids_tmp = tokenizer(prompt(queries[idx]), padding="max_length",
                                        return_tensors="np",
                                        max_length=args.pad_to_seq_len).input_ids
            logger.info(input_ids_tmp.shape)
            output_ids_tmp = model.generate(input_ids_tmp,
                                            do_sample=True,
                                            temperature=args.temperature,
                                            max_new_tokens=args.max_tokens,
                                            stop=args.stop_token)
            input_strs = tokenizer.batch_decode(input_ids_tmp, skip_special_tokens=True)
            output_strs = tokenizer.batch_decode(output_ids_tmp, skip_special_tokens=True)
            anwsers = [ output_strs[i][len(input_strs[i]):] for i in range(len(input_strs))]
            logger.info(f"====> {anwsers[0]} <====")
            preds.extend(anwsers)
            idx += 1
            logger.info(f"Current Inference query elapsed: {time.time() - tic:.2f} s")
        # Save trial predictions
        save_data = test_data.iloc[:args.num_print].copy(deep=True).reset_index()
        gt = gt[:args.num_print]
        save_data["preds"] = preds
        save_data["queries"] = queries[:args.num_print]

        prec, rec, acc, f1 = compute_metrics(preds, gt, task)

        logger.info(
            f"Metrics Trial {trial_num}\n"
            f"Prec: {prec:.3f} Recall: {rec:.3f} Acc: {acc:.3f} F1: {f1:.3f}"
        )
        trial_metrics["rec"].append(rec)
        trial_metrics["prec"].append(prec)
        trial_metrics["acc"].append(acc)
        trial_metrics["f1"].append(f1)

        output_file = (
            Path(args.output_dir)
            / f"{Path(args.data_dir).stem}"
            / f"{test_file}"
            / f"{args.run_tag}"
            / f"{args.k}k"
            f"_{int(args.add_task_instruction)}inst"
            f"_{int(args.class_balanced)}cb"
            f"_{args.sample_method}"
            f"_{args.model}"
            f"_{args.num_print}run"
            f"_{int(args.dry_run)}dry" / f"trial_{trial_num}.feather"
        )

        output_file.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saved to {output_file}")

        save_data.to_feather(output_file)

    for k, values in list(trial_metrics.items()):
        trial_metrics[f"{k}_avg"] = np.average(values)
        trial_metrics[f"{k}_std"] = np.std(values)

    output_metrics = output_file.parent / "metrics.json"
    json.dump(trial_metrics, open(output_metrics, "w"))

    logger.info(f"Final Metrics {json.dumps(trial_metrics, indent=4)}")
    logger.info(f"Metrics dumped to {output_metrics}")
    # Shutdown
    logger.info("Shutdown FlexGen...")
    env.close_copy_threads()


def batch_query_test(args, task_instruction, test_data, task, pd_data_files, test_file):
    # Initialize environment
    tokenizer = get_tokenizer(args.model)
    env = ExecutionEnv.create(args.offload_dir)

    # Offloading policy
    policy = Policy(args.gpu_batch_size, args.num_gpu_batches,
                    args.percent[0], args.percent[1],
                    args.percent[2], args.percent[3],
                    args.percent[4], args.percent[5],
                    overlap=True, sep_layer=True, pin_weight=args.pin_weight,
                    cpu_cache_compute=args.cpu_cache_compute, attn_sparsity=1.0,
                    compress_weight=args.compress_weight,
                    comp_weight_config=CompressionConfig(
                        num_bits=4, group_size=64,
                        group_dim=0, symmetric=False),
                    compress_cache=args.compress_cache,
                    comp_cache_config=CompressionConfig(
                        num_bits=4, group_size=64,
                        group_dim=2, symmetric=False))

    logger.info(f"Init weights begin.")
    tic = time.time()
    model = OptLM(args.model, env, args.path, policy)
    logger.info(f"Init weights end. Elapsed: {time.time() - tic:.2f} s.")

    if args.add_task_instruction:
        prompt = lambda x: f"{task_instruction} {x}"
    else:
        prompt = lambda x: f"{x}"
    trial_metrics = {"prec": [], "rec": [], "f1": [], "acc": [], "total_time": [],
                     "output_throughput": [], "total_throughput": []}

    saved_prefix = None
    
    for trial_num in range(args.num_trials):
        np.random.seed(args.seed + trial_num)
        queries = []
        for _, row in test_data.iterrows():
            serialized_r = row["text"]
            if args.sample_method == "manual":
                prefix_exs = prompt_utils.get_manual_prompt(args.data_dir, row)
            elif args.sample_method == "validation_clusters":
                if saved_prefix is None:
                    logger.info("Generating validation cluster prompt.")
                    saved_prefix = prompt_utils.get_validation_prompt(
                        args.validation_path,
                        num_examples=args.k,
                        task=task,
                    )
                prefix_exs = saved_prefix
            else:
                if saved_prefix is None:
                    saved_prefix = prompt_utils.get_random_prompt(
                        pd_data_files["train"], num_examples=args.k
                    )
                prefix_exs = saved_prefix
            queries.append((prefix_exs + "\n" + serialized_r).strip())

        gt = test_data["label_str"]
        preds = []
        idx = 0
        
        max_prompt_seq_length = 0
        prompt_strs = []
        for _ in range(args.num_run):
            # if idx == 0:
            #    logger.info(f"This is a sample prompt: {prompt(queries[idx])}")
            prompt_strs.append(prompt(queries[idx]))
            
            current_prompt_tmp = tokenizer(prompt(queries[idx]), padding="max_length",
                                      return_tensors="np", max_length=args.pad_to_seq_len).input_ids
            # logger.info(f"Current prompt <{idx}> length: {current_prompt_tmp.shape[1]}")
            max_prompt_seq_length = max(max_prompt_seq_length, current_prompt_tmp.shape[1])
            idx += 1
        
        logger.info(f"max_prompt_seq_length: {max_prompt_seq_length}")
        tic = time.time()
        
        input_ids = tokenizer(prompt_strs, padding="max_length",
                                  return_tensors="np",
                                  max_length=max_prompt_seq_length).input_ids
        output_ids = []
        
        flexgen_batch_size = args.gpu_batch_size*args.num_gpu_batches
        num_batched_run = math.floor(args.num_run/flexgen_batch_size)
        args.num_run = num_batched_run * flexgen_batch_size
        input_ids = input_ids[0:args.num_run]
        
        for i in tqdm(range(num_batched_run)):
            input_ids_tmp = input_ids[i*flexgen_batch_size: (i+1)*flexgen_batch_size]
            output_ids_tmp = model.generate(input_ids_tmp,
                                            do_sample=True,
                                            temperature=args.temperature,
                                            max_new_tokens=args.max_tokens,
                                            stop=args.stop_token)
            output_ids.extend(output_ids_tmp)
        
        toc = time.time()
        input_strs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        output_strs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [output_strs[i][len(input_strs[i]):] for i in range(len(input_strs))]
        
        total_time = time.time() - tic
        total_prompt_tokens = args.num_run * max_prompt_seq_length
        total_generate_tokens = args.num_run * args.max_tokens
        output_throughput = total_generate_tokens/total_time
        total_throughput = (total_prompt_tokens+total_generate_tokens)/total_time
        logger.info(f"Batch inference run end. Elapsed: {total_time:.2f} s;")
        logger.info(f"Output throughput: {output_throughput:.2f} token/s;")
        logger.info(f"Total throughput: {total_throughput:.2f} token/s;")
        # Save trial predictions
        save_data = test_data.iloc[:args.num_run].copy(deep=True).reset_index()
        gt = gt[:args.num_run]
        save_data["preds"] = preds
        save_data["queries"] = queries[:args.num_run]

        prec, rec, acc, f1 = compute_metrics(preds, gt, task)

        logger.info(
            f"Metrics Trial {trial_num}\n"
            f"Prec: {prec:.3f} Recall: {rec:.3f} Acc: {acc:.3f} F1: {f1:.3f} \n"
            f"<FlexGen> time: {total_time:.3f} \n" 
            f"<FlexGen> output throughput: {output_throughput:.3f} \n"
            f"<FlexGen> total throughput: {total_throughput:.3f}"
        )
        trial_metrics["rec"].append(rec)
        trial_metrics["prec"].append(prec)
        trial_metrics["acc"].append(acc)
        trial_metrics["f1"].append(f1)
        trial_metrics["total_time"].append(total_time)
        trial_metrics["output_throughput"].append(output_throughput)
        trial_metrics["total_throughput"].append(total_throughput)

        output_file = (
            Path(args.output_dir)
            / f"{Path(args.data_dir).stem}"
            / f"{test_file}"
            / f"{args.run_tag}"
            / f"{args.k}k"
            f"_{int(args.add_task_instruction)}inst"
            f"_{int(args.class_balanced)}cb"
            f"_{args.sample_method}"
            f"_{args.model}"
            f"_{args.num_run}run"
            f"_{int(args.dry_run)}dry" / f"trial_{trial_num}.feather"
        )

        output_file.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saved to {output_file}")

        save_data.to_feather(output_file)


    for k, values in list(trial_metrics.items()):
        trial_metrics[f"{k}_avg"] = np.average(values)
        trial_metrics[f"{k}_std"] = np.std(values)

    output_metrics = output_file.parent / "metrics.json"
    json.dump(trial_metrics, open(output_metrics, "w"))

    logger.info(f"Final Metrics {json.dumps(trial_metrics, indent=4)}")
    logger.info(f"Metrics dumped to {output_metrics}")
    # Shutdown
    logger.info("Shutdown FlexGen...")
    env.close_copy_threads()
    

def main():
    """Run main method."""
    args = parse_args()
    if args.num_trials < 1:
        raise ValueError("num_trials must be greater than 0.")
    # Get absolute path
    args.data_dir = str(Path(args.data_dir).resolve())
    setup_logger(args.output_dir)
    logger.info(json.dumps(vars(args), indent=4))

    # Will set seed for pandas
    np.random.seed(args.seed)

    test_file = "test" if args.do_test else "validation"

    # Read pandas DF datasets
    pd_data_files = data_utils.read_data(
        data_dir=args.data_dir,
        class_balanced=args.class_balanced,
        add_instruction=False,
        max_train_samples=-1,
        max_train_percent=-1,
        sep_tok=args.sep_tok,
        nan_tok=args.nan_tok,
    )
    if test_file not in pd_data_files:
        raise ValueError(f"Need {test_file} data")

    train_data = pd_data_files["train"]
    test_data = pd_data_files[test_file]
    task = constants.DATA2TASK[args.data_dir]
    logger.info(f"Using {args.task_instruction_idx} instruction idx")
    task_instruction = constants.DATA2INSTRUCT[args.data_dir]
    num_run = args.num_run
    if args.num_run == -1:
        num_run = test_data.shape[0]
    num_run = min(num_run, test_data.shape[0])

    logger.info(f"Train shape is {train_data.shape[0]}")
    logger.info(f"Test shape is {test_data.shape[0]}")
    logger.info(f"Running {num_run} examples for {args.num_trials} trials.")
    
    if args.batch_run:
        logger.info("Call batch_query_test")
        batch_query_test(args, task_instruction, test_data, task, pd_data_files, test_file)
    else:
        logger.info("Call single_query_test")
        single_query_test(args, task_instruction, test_data, task, pd_data_files, test_file)


if __name__ == "__main__":
    main()
