import time
from argparse import ArgumentParser
from statistics import mean

import torch
from petals import DistributedBloomConfig, DistributedBloomForCausalLM
from torch.multiprocessing import Process, Event, Queue
from transformers import AutoTokenizer, BloomConfig, OPTConfig


def _patch_bloom_config(bloom_config: BloomConfig, opt_config: OPTConfig):
    bloom_config.hidden_size = opt_config.hidden_size
    bloom_config.n_head = opt_config.num_attention_heads
    bloom_config.n_layer = opt_config.num_hidden_layers
    bloom_config.vocab_size = opt_config.vocab_size


def client_process(
    finished_warmup,
    can_start,
    config_bloom,
    num_micro_batches,
    batch_size,
    sequence_length,
    max_tokens,
    process_index,
    queue: Queue,
) -> None:
    torch.set_num_threads(1)
    torch.cuda.set_device(process_index % torch.cuda.device_count())

    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b")
    inputs = torch.randint(0, tokenizer.vocab_size, size=(batch_size, sequence_length), device="cuda")

    model = DistributedBloomForCausalLM(config_bloom)
    model.cuda()

    # warmup
    model.generate(inputs, max_new_tokens=1, do_sample=False)
    finished_warmup.set()
    can_start.wait()

    for _ in range(num_micro_batches):
        start = time.monotonic()
        model.generate(inputs, max_new_tokens=max_tokens, do_sample=False)
        end = time.monotonic()
        queue.put(end - start)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--initial_peers",
        nargs="*",
        help="Multiaddrs of the peers that will welcome you into the existing DHT. "
             "Example: /ip4/203.0.113.1/tcp/31337/p2p/XXXX /ip4/203.0.113.2/tcp/7777/p2p/YYYY",
    )
    parser.add_argument('--prefix', default="facebook/opt-175b", help="Prefix of the model.")
    parser.add_argument('--batch-size', "-b", default=1, type=int)
    parser.add_argument('--num-micro-batches', default=1, type=int)
    parser.add_argument('--num-processes', default=1, type=int)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    config_bloom = DistributedBloomConfig.from_pretrained("bigscience/bloom-petals")
    config_bloom.initial_peers = args.initial_peers

    if args.prefix == "facebook/opt-6.7b":
        config_bloom.dht_prefix = "opt6b"
    else:
        config_bloom.dht_prefix = args.prefix

    if args.prefix == "facebook/opt-175b":
        config_bloom.hidden_size = 12288
        config_bloom.n_layer = 96
        config_bloom.n_head = 96
        config_bloom.vocab_size = 50272
    else:
        config_opt = OPTConfig.from_pretrained(args.prefix)
        _patch_bloom_config(config_bloom, config_opt)

    for sequence_length in 256, 512, 1024:
        run_bench(args=args, sequence_length=sequence_length, max_tokens=32, config_bloom=config_bloom)

    if args.prefix == "facebook/opt-30b":
        for max_tokens in range(33):
            run_bench(args=args, sequence_length=512, max_tokens=max_tokens, config_bloom=config_bloom)


def run_bench(args, sequence_length, max_tokens, config_bloom):
    queue = Queue()
    processes = []
    warmup_events = []
    can_start = Event()
    for i in range(args.num_processes):
        print("create process", i)
        warmup_event = Event()
        proc = Process(target=client_process,
                       args=(warmup_event, can_start, config_bloom, args.num_micro_batches, args.batch_size,
                             sequence_length, max_tokens, i, queue)
                       )
        proc.start()
        processes.append(proc)
        warmup_events.append(warmup_event)
    for event in warmup_events:
        event.wait()
    can_start.set()
    start = time.monotonic()
    for i, proc in enumerate(processes):
        print("join process", i)
        proc.join()
    end = time.monotonic()
    latencies = []
    while not queue.empty():
        latencies.append(queue.get())
    print("total time", end - start)
    total_tokens = args.batch_size * args.num_micro_batches * args.num_processes * max_tokens
    print("total tokens", total_tokens)

    throughput = total_tokens / (end - start)
    print("throughput", throughput)
    latency = mean(latencies)
    print("average latency", latency)
    with open(args.output, "a") as f:
        print("\t".join(
            map(str,
                [args.batch_size, args.num_micro_batches, args.num_processes, sequence_length, max_tokens,
                 throughput, latency]
                )), file=f)


if __name__ == "__main__":
    main()
