"""
Run a scenario from HELM.

See also: https://crfm.stanford.edu/helm/
helm package version: 0.2.1
"""
import argparse
from dataclasses import asdict, replace
import json
import math
import os
import time

from flexgen.flex_opt import (Policy, OptLM, ExecutionEnv, CompressionConfig,
        str2bool)
from helm.benchmark.presentation.run_entry import RunEntry
from helm.benchmark.run import run_entries_to_run_specs
from helm.benchmark.run_specs import (ScenarioSpec, RunSpec, get_summarization_adapter_spec,
    get_summarization_metric_specs, get_generative_harms_metric_specs,
    ADAPT_MULTIPLE_CHOICE_JOINT, get_multiple_choice_adapter_spec)
from helm.benchmark.runner import (create_scenario, AdapterFactory, with_instance_ids, create_metric,
    TokensMetric, Metric, MetricSpec, MetricResult, PerInstanceStats, create_metric, Stat,
    ScenarioState, Counter, MetricName, ensure_directory_exists, write, asdict_without_nones,
    DataPreprocessor)
from helm.common.request import Request, RequestResult, Sequence, Token
from helm.common.tokenization_request import (TokenizationRequestResult,
    TokenizationRequest, TokenizationToken, DecodeRequest, DecodeRequestResult)
from helm.proxy.clients.client import truncate_sequence
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig


class OptTokenizer:
    # Adapted from helm/proxy/clients/huggingface_client.py

    def __init__(self, name):
        self.tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
        self.tokenizer.add_bos_token = False
        if 'galactica' in name:
            config = AutoConfig.from_pretrained(name)
            self.tokenizer.pad_token = config.pad_token_id
            self.tokenizer.eos_token = config.eos_token_id

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        tokenizer = self.tokenizer

        def do_it():
            if request.encode:
                if request.truncation:
                    tokens = tokenizer.encode(
                        request.text,
                        truncation=request.truncation,
                        max_length=request.max_length,
                        add_special_tokens=False,
                    )
                else:
                    tokens = tokenizer.encode(request.text, add_special_tokens=False)
            else:
                tokens = tokenizer.tokenize(request.text)
            return {"tokens": tokens}

        result = do_it()

        return TokenizationRequestResult(
            success=True,
            cached=False,
            text=request.text,
            tokens=[TokenizationToken(value) for value in result["tokens"]],
            request_time=0,
        )

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        tokenizer = self.tokenizer


        def do_it():
            return {
                "text": tokenizer.decode(
                    request.tokens, clean_up_tokenization_spaces=request.clean_up_tokenization_spaces
                )
            }

        result = do_it()

        return DecodeRequestResult(
            success=True, cached=False, text=result["text"], request_time=0,
        )


def get_hf_generation_args(request, tokenizer):
    # Adapted from helm/proxy/clients/huggingface_client.py
    raw_request = {
        "engine": request.model_engine,
        "prompt": request.prompt,
        "temperature": 1e-7 if request.temperature == 0 else request.temperature,
        "num_return_sequences": request.num_completions,
        "max_new_tokens": request.max_tokens,
        "top_p": request.top_p,
        "echo_prompt": request.echo_prompt,
        "top_k_per_token": request.top_k_per_token,
        "stop_sequences": request.stop_sequences,
    }

    raw_request["do_sample"] = True
    raw_request["return_dict_in_generate"] = True
    raw_request["output_scores"] = True
    top_k_per_token: int = raw_request["top_k_per_token"]
    del raw_request["top_k_per_token"]

    if len(raw_request["stop_sequences"]) > 0:
        stop_sequence_ids = tokenizer(raw_request["stop_sequences"])
        # Total number of stop words should be 1.
        assert len(stop_sequence_ids.input_ids) == 1
        # Total number of tokens in each stop word should be 1.
        assert len(stop_sequence_ids.input_ids[0]) == 1
        del raw_request["stop_sequences"]
        raw_request["eos_token_id"] = stop_sequence_ids.input_ids[0][0]

    # Strip out irrelevant parameters
    relevant_raw_request = {
        key: raw_request[key]
        for key in raw_request
        if key not in ["engine", "prompt", "echo_prompt", "stop_sequences"]
    }
    print(f"relevant_raw_request: {relevant_raw_request}")

    return relevant_raw_request


def get_batches(scenario_state, tokenizer, batch_size, pad_to_seq_len):
    prompts = []
    for r in scenario_state.request_states:
        prompts.append(r.request.prompt)

    # Tokenize
    input_ids = tokenizer(prompts, padding="max_length",
                          return_tensors="np",
                          max_length=pad_to_seq_len).input_ids
    assert len(input_ids.shape) == 2, f"Please use a longer pad_to_seq_len. current = {pad_to_seq_len}"
    max_seq_len = max(np.sum(input_ids != tokenizer.pad_token_id, axis=1))
    if max_seq_len != pad_to_seq_len:
        pad_to_seq_len = min(int(math.ceil(max_seq_len / 256) * 256), pad_to_seq_len)
        input_ids = tokenizer(prompts, padding="max_length",
                              return_tensors="np",
                              max_length=pad_to_seq_len).input_ids
        assert len(input_ids.shape) == 2, f"Auto-adjusting pad_to_seq_len failed. current = {pad_to_seq_len}"
        max_seq_len = max(np.sum(input_ids != tokenizer.pad_token_id, axis=1))

    print(f"Max sequence length: {max_seq_len}, Pad to sequences length: {pad_to_seq_len}")

    # Pad and divide into batches
    n_prompts = len(prompts)
    if n_prompts % batch_size != 0:
        input_ids = np.concatenate((input_ids, np.full((batch_size - n_prompts % batch_size,
            input_ids.shape[1]), tokenizer.pad_token_id, dtype=input_ids.dtype)))

    num_batches = len(input_ids) // batch_size
    assert len(input_ids) % batch_size == 0
    return [
        {"input_ids": input_ids[i * batch_size: (i+1) * batch_size]}
        for i in range(num_batches)
    ]


def execute(scenario_state, tokenizer, effective_bs, pad_to_seq_len):
    generation_args = get_hf_generation_args(
        scenario_state.request_states[0].request, tokenizer)
    batches = get_batches(scenario_state, tokenizer,
                          effective_bs, pad_to_seq_len=pad_to_seq_len)

    # Initialize environment
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

    print(f"Init weights begin.")
    tic = time.time()
    model = OptLM(args.model, env, args.path, policy)
    print(f"Init weights end. Elapsed: {time.time() - tic:.2f} s", flush=True)

    # Generate
    print(f"Generate begin. #sequences: {len(batches) * effective_bs}")
    tic = time.time()
    input_ids_batches = []
    output_ids_batches = []
    for batch in tqdm(batches):
        input_ids_tmp = batch["input_ids"]
        output_ids_tmp = model.generate(
            input_ids_tmp,
            do_sample=generation_args["do_sample"],
            temperature=generation_args["temperature"],
            max_new_tokens=generation_args["max_new_tokens"],
            stop=generation_args.get("eos_token_id", None))
        input_ids_batches.append(input_ids_tmp)
        output_ids_batches.append(output_ids_tmp)
    print(f"Generate end. Elapsed: {time.time() - tic:.2f} s", flush=True)

    input_ids = np.concatenate(input_ids_batches)
    output_ids = np.concatenate(output_ids_batches)
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    #print("Outputs:\n" + 70 * '-')
    ##for i in range(len(outputs)):
    #for i in [0, len(outputs) - 1]:
    #    print(f"{i}:\n{outputs[i]}")
    #    print("-" * 70)

    # Shutdown
    print("Shutdown...")
    env.close_copy_threads()

    request_states = []
    for i, request_state in enumerate(scenario_state.request_states):
        request = request_state.request
        encoded_input = input_ids[i]
        sequences = [output_ids[i]]
        if not request.echo_prompt:
            sequences = [sequence[len(encoded_input) :] for sequence in sequences]

        all_tokens = [tokenizer.convert_ids_to_tokens(sequence) for sequence in sequences]
        all_decoded_text = tokenizer.batch_decode(sequences)
        all_logprobs_of_chosen_tokens = [[0] * len(x) for x in all_tokens]
        all_top_logprobs_dicts = [[{}] * len(x) for x in all_tokens]

        completions = []
        for (decoded_text, tokens, logprobs_of_chosen_tokens, top_logprobs_dicts) in zip(
            all_decoded_text, all_tokens, all_logprobs_of_chosen_tokens, all_top_logprobs_dicts
        ):
            completions.append(
                {
                    "text": decoded_text,
                    "tokens": tokens,
                    "logprobs": logprobs_of_chosen_tokens,
                    "top_logprobs_dicts": top_logprobs_dicts,
                }
            )
        response = {
            "completions": completions, "input_length": len(encoded_input)}

        completions = []
        for raw_completion in response["completions"]:
            sequence_logprob: float = 0
            tokens: List[Token] = []

            if request.echo_prompt:
                # Add prompt to list of generated tokens.
                generated_tokens = raw_completion["tokens"][response["input_length"] :]
                for token_text in raw_completion["tokens"][: response["input_length"]]:
                    tokens.append(Token(text=token_text, logprob=0.0, top_logprobs={}))
            else:
                generated_tokens = raw_completion["tokens"]

            # Compute logprob for the entire sequence.
            for token_text, logprob, top_logprobs_dict in zip(
                generated_tokens, raw_completion["logprobs"], raw_completion["top_logprobs_dicts"]
            ):
                tokens.append(Token(text=token_text, logprob=logprob, top_logprobs=top_logprobs_dict))
                sequence_logprob += logprob

            completion = Sequence(text=raw_completion["text"], logprob=sequence_logprob, tokens=tokens)
            completion = truncate_sequence(completion, request)
            completions.append(completion)

        result = RequestResult(
            success=True,
            cached=False,
            request_time=0,
            request_datetime=0,
            completions=completions,
            embedding=[],
        )

        request_states.append(replace(request_state, result=result))

    return ScenarioState(scenario_state.adapter_spec, request_states)


def run_entry(description, pad_to_seq_len, args):
    effective_bs = args.gpu_batch_size * args.num_gpu_batches
    parallelism = 4

    ##### RunSpec #####
    run_entries = [RunEntry(description, priority=1, groups=None)]
    run_specs = run_entries_to_run_specs(
        run_entries=run_entries,
        max_eval_instances=args.max_eval_instances,
        num_train_trials=3,
    )
    run_spec = run_specs[0]
    run_path: str = os.path.join(args.run_path, run_spec.name)
    ensure_directory_exists(run_path)
    eval_cache_path: str = os.path.join(run_path, "eval_cache")
    ensure_directory_exists(eval_cache_path)

    ##### Adapter #####
    #tokenizer_service = OptTokenizer("facebook/opt-30b")
    tokenizer_service = OptTokenizer(args.model)
    tokenizer = tokenizer_service.tokenizer
    adapter = AdapterFactory.get_adapter(run_spec.adapter_spec, tokenizer_service)

    ##### Scenario #####
    print(run_spec)
    scenario = create_scenario(run_spec.scenario_spec)
    scenario.output_path = f"data/{run_spec.name}"
    os.makedirs(scenario.output_path, exist_ok=True)
    instances = scenario.get_instances()

    # Give each instance a unique ID
    instances = with_instance_ids(instances)

    # Get the instances necessary for this run.
    instances = adapter.get_run_instances(instances)

    # Data preprocessing
    instances = DataPreprocessor(run_spec.data_augmenter_spec).preprocess(
        instances, parallelism=parallelism
    )
    scenario_state = adapter.adapt(instances, parallelism=parallelism)

    ##### Execute #####
    if pad_to_seq_len is None:
        pad_to_seq_len = adapter.window_service.max_sequence_length - run_spec.adapter_spec.max_tokens + 1
    scenario_state = execute(scenario_state, tokenizer, effective_bs, pad_to_seq_len)

    ##### Metrics #####
    metrics = (
        [create_metric(metric_spec) for metric_spec in run_spec.metric_specs]) + [TokensMetric()]
    metrics = [metrics[0]]

    stats: List[Stat] = []
    per_instance_stats: List[PerInstanceStats] = []
    for metric in metrics:
        metric_result: MetricResult = metric.evaluate(
            scenario_state,
            tokenizer_service,
            eval_cache_path,
            parallelism=parallelism,
        )
        stats.extend(metric_result.aggregated_stats)
        per_instance_stats.extend(metric_result.per_instance_stats)

    # Check that there aren't duplicate `Stat`s
    # Note: doesn't catch near misses.
    metric_counts: typing.Counter[MetricName] = Counter([stat.name for stat in stats])
    for metric_name, count in metric_counts.items():
        if count > 1:
            print(f"WARNING: duplicate metric name {metric_name}")

    # Print out the number of stats
    print(f"Generated {len(stats)} stats.")

    # Output benchmarking information and results to files
    write(os.path.join(run_path, "run_spec.json"), json.dumps(asdict_without_nones(run_spec), indent=2))

    # Write out scenario
    write(os.path.join(run_path, "scenario.json"), json.dumps(asdict_without_nones(scenario), indent=2))

    # Write scenario state
    write(os.path.join(run_path, "scenario_state.json"), json.dumps(asdict_without_nones(scenario_state), indent=2))

    write(
        os.path.join(run_path, "stats.json"), json.dumps([asdict_without_nones(stat) for stat in stats], indent=2)
    )
    write(
        os.path.join(run_path, "per_instance_stats.json"),
        json.dumps(list(map(asdict_without_nones, per_instance_stats)), indent=2),
    )


def main(args):
    run_entry(args.description, args.pad_to_seq_len, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--description", type=str, required=True)
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
    parser.add_argument("--max-eval-instances", type=int)
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
    args = parser.parse_args()

    assert len(args.percent) == 6

    main(args)
