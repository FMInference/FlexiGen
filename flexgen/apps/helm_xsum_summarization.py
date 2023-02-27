"""
Run the text summarization scenario from helm.

See also: https://crfm.stanford.edu/helm/
"""
import argparse
from dataclasses import asdict

import helm
from helm.benchmark.run_specs import (ScenarioSpec, RunSpec, get_summarization_adapter_spec,
    get_summarization_metric_specs, get_generative_harms_metric_specs)
from helm.benchmark.runner import (create_scenario, AdapterFactory, with_instance_ids)
from helm.common.tokenization_request import (TokenizationRequestResult,
    TokenizationRequest, TokenizationToken)
from flexgen.flex_opt import (Policy, OptLM, ExecutionEnv, CompressionConfig,
        str2bool)
from transformers import AutoTokenizer


def get_xsum_sampled_summarization_spec(temperature: float = 0.3, device: str = "cpu",
        max_eval_instances: int = 512) -> RunSpec:
    # Adapted from helm/benchmark/run_specs.py
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.summarization_scenario.SummarizationScenario",
        args={
            "dataset_name": "xsum-sampled",
            "sampling_min_length": 50,
            "sampling_max_length": 150,
            "doc_max_length": 512,
        },
    )

    adapter_spec = get_summarization_adapter_spec(
        num_sents=1,
        max_tokens=64,  # From Zhang et al. 2020 (https://arxiv.org/pdf/1912.08777.pdf)
        temperature=temperature,  # The default of 0.3 was determined in initial pilots, comparing to 0.7 and 1.0
        max_eval_instances=max_eval_instances,
        model="huggingface/gpt2",
    )

    return RunSpec(
        name=f"summarization_xsum:temperature={temperature},device={device}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_summarization_metric_specs({"task": "summarization_xsum_sampled", "device": device})
        + get_generative_harms_metric_specs(),
        groups=["summarization_xsum"],
    )


class OptTokenizer:
    # Adapted from helm/proxy/clients/huggingface_client.py

    def __init__(self, name):
        self.tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
        self.tokenizer.add_bos_token = False

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


def get_hf_generation_args(request, tokenizer):
    # Adapted from huggingface_client.py
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

    return relevant_raw_request


def get_batches(scenario_state, tokenizer, batch_size, max_seq_length):
    prompts = []
    for r in scenario_state.request_states:
        prompts.append(r.request.prompt)

    input_ids = tokenizer(prompts, padding="max_length",
                          return_tensors="np",
                          max_length=max_seq_length).input_ids
    return [
        {"input_ids": input_ids},
    ]


def main(args):
    effective_bs = args.gpu_batch_size * args.num_gpu_batches
    max_eval_instances = effective_bs

    run_spec = get_xsum_sampled_summarization_spec(max_eval_instances=max_eval_instances)
    tokenizer_service = OptTokenizer("facebook/opt-30b")
    tokenizer = tokenizer_service.tokenizer
    adapter = AdapterFactory.get_adapter(run_spec.adapter_spec, tokenizer_service)
    scenario = create_scenario(run_spec.scenario_spec)
    instances = scenario.get_instances()
    instances = with_instance_ids(instances)
    instances = adapter.get_run_instances(instances)
    scenario_state = adapter.adapt(instances, parallelism=1)

    generation_args = get_hf_generation_args(
        scenario_state.request_states[0].request, tokenizer)
    batches = get_batches(scenario_state, tokenizer,
                          effective_bs, max_seq_length=1024)
    input_ids = batches[0]["input_ids"]

    # Initialize environment
    env = ExecutionEnv.create(args.offload_dir)

    # Offloading policy
    policy = Policy(args.gpu_batch_size, args.num_gpu_batches,
                    args.percent[0], args.percent[1],
                    args.percent[2], args.percent[3],
                    args.percent[4], args.percent[5],
                    overlap=True, sep_layer=True, pin_weight=args.pin_weight,
                    cpu_cache_compute=False, attn_sparsity=1.0,
                    compress_weight=args.compress_weight,
                    comp_weight_config=CompressionConfig(
                        num_bits=4, group_size=64,
                        group_dim=0, symmetric=False),
                    compress_cache=args.compress_cache,
                    comp_cache_config=CompressionConfig(
                        num_bits=4, group_size=64,
                        group_dim=2, symmetric=False))

    model = OptLM(args.model, env, args.path, policy)

    # Generate
    print("Generate...")
    output_ids = model.generate(
        input_ids,
        do_sample=generation_args["do_sample"],
        temperature=generation_args["temperature"],
        max_new_tokens=generation_args["max_new_tokens"],
        stop=generation_args["eos_token_id"])
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    print("Outputs:\n" + 70 * '-')
    for i in range(len(outputs)):
        print(f"{i}:\n{outputs[i]}")
        print("-" * 70)

    # Shutdown
    print("Shutdown...")
    env.close_copy_threads()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-6.7b",
        help="The model name.")
    parser.add_argument("--path", type=str, default="~/opt_weights",
        help="The path to the model weights. If there are no cached weights, "
             "FlexGen will automatically download them from HuggingFace.")
    parser.add_argument("--offload-dir", type=str, default="~/flexgen_offload_dir",
        help="The directory to offload tensors. ")
    parser.add_argument("--gpu-batch-size", type=int, default=4)
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
    parser.add_argument("--compress-weight", action="store_true",
        help="Whether to compress weight.")
    parser.add_argument("--compress-cache", action="store_true",
        help="Whether to compress cache.")
    args = parser.parse_args()

    assert len(args.percent) == 6

    main(args)
