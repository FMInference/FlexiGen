import argparse
from dataclasses import dataclass

from flexgen.utils import run_cmd


@dataclass
class Case:
    command: str
    name: str = ""
    use_page_maga: bool = False


suite_1b3_test = [
    # All GPU
    Case("--model facebook/opt-1.3b --gpu-batch-size 16 --percent 100 0 100 0 100 0 --cut-gen-len 8", "All GPU"),
    # Weight on CPU, cache on GPU
    Case("--model facebook/opt-1.3b --gpu-batch-size 16 --percent 0 100 100 0 100 0 --cut-gen-len 8", "Weight on CPU, cache on GPU"),
    # Weight on GPU, cache on CPU
    Case("--model facebook/opt-1.3b --gpu-batch-size 16 --percent 100 0 0 100 100 0 --cut-gen-len 8 --cpu", "Weight on GPU, cache on CPU"),
    # Weight on CPU, cache on CPU
    Case("--model facebook/opt-1.3b --gpu-batch-size 16 --percent 0 100 0 100 100 0 --cut-gen-len 8 --cpu", "Weight on CPU, cache on CPU"),
    # Weight on disk, cache on GPU
    Case("--model facebook/opt-1.3b --gpu-batch-size 16 --percent 0 0 100 0 100 0 --cut-gen-len 8", "Weight on disk, cache on GPU", True),
    # Weight on GPU, cache on disk
    Case("--model facebook/opt-1.3b --gpu-batch-size 16 --percent 100 0 0 0 100 0 --cut-gen-len 8 --cpu", "Weight on GPU, cache on disk", True),
    # Weight on CPU/GPU (50-50 split), cache on GPU
    Case("--model facebook/opt-1.3b --gpu-batch-size 16 --percent 50 50 100 0 100 0 --cut-gen-len 8", "Weight on both CPU/GPU (50-50 split), cache on GPU"),
    # Weight on GPU, cache on CPU/GPU (50-50 split)
    Case("--model facebook/opt-1.3b --gpu-batch-size 16 --percent 100 0 50 50 100 0 --cut-gen-len 8 --cpu", "Weight on GPU, cache on CPU/GPU (50-50 split)"),
    # Weight on GPU, cache on disk, sparse attention
    Case("--model facebook/opt-1.3b --gpu-batch-size 16 --percent 100 0 0 0 100 0 --cut-gen-len 8 --cpu --attn-sparsity 0.1", "Weight on GPU, cache on disk, sparse attention", True),
    # Weight on GPU, cache on disk, cache quantization
    Case("--model facebook/opt-1.3b --gpu-batch-size 16 --percent 100 0 0 0 100 0 --cut-gen-len 8 --compress-cache", "Weight on GPU, cache on disk, cache quantization", True),
    # All GPU, 2 GPU batches
    Case("--model facebook/opt-1.3b --gpu-batch-size 16 --percent 100 0 100 0 100 0 --cut-gen-len 8 --num-gpu-batches 2", "All GPU, 2 gpu batches"),
]

suite_6b7_1x1 = [
    # seq_len = 256, gen_len = 32
    # 53.29 token/s
    Case("--model facebook/opt-6.7b --path _DUMMY_ --prompt-len 256 --gen-len 32 --percent 100 0 100 0 100 0 --gpu-batch-size 4 --overlap False"),
    # seq_len = 512, gen_len = 32
    Case("--model facebook/opt-6.7b --path _DUMMY_ --percent 100 0 100 0 100 0 --gpu-batch-size 2 --overlap False"),
    # seq_len = 1024, gen_len = 32
    Case("--model facebook/opt-6.7b --path _DUMMY_ --percent 100 0 100 0 100 0 --gpu-batch-size 1 --overlap False --prompt-len 1024"),
]

suite_6b7_1x1_comp = [
    # seq_len = 256, gen_len = 32
    # 56.72 token/s
    Case("--model facebook/opt-6.7b --path _DUMMY_ --prompt-len 256 --gen-len 32 --percent 100 0 100 0 100 0 --gpu-batch-size 128 --overlap False --compress-weight --compress-cache"),
    # seq_len = 512, gen_len = 32
    Case("--model facebook/opt-6.7b --path _DUMMY_ --percent 100 0 100 0 100 0 --gpu-batch-size 72 --overlap False --compress-weight --compress-cache"),
    # seq_len = 1024, gen_len = 32
    Case("--model facebook/opt-6.7b --path _DUMMY_ --percent 100 0 100 0 100 0 --gpu-batch-size 28 --overlap False --compress-weight --compress-cache --prompt-len 1024"),
]

suite_30b_1x1 = [
    # seq_len = 256, gen_len = 32
    # 16.01 token/s
    Case("--model facebook/opt-30b --path _DUMMY_ --prompt-len 256 --gen-len 32 --percent 10 90 0 100 0 100 --gpu-batch-size 160 --num-gpu-batches 2 --cpu --debug fewer_batch", "", False),
    # seq_len = 512, gen_len = 32
    Case("--model facebook/opt-30b --path _DUMMY_ --percent 20 80 0 100 0 100 --gpu-batch-size 48 --num-gpu-batches 3 --cpu --debug fewer_batch"),
    # seq_len = 1024, gen_len = 32
    Case("--model facebook/opt-30b --path _DUMMY_ --percent 4 96 0 100 0 100 --gpu-batch-size 20 --num-gpu-batches 4 --cpu --debug fewer_batch --prompt-len 1024"),
]

suite_30b_1x1_comp = [
    # seq_len = 256, gen_len = 32
    # 16.86 token/s
    Case("--model facebook/opt-30b --path _DUMMY_ --prompt-len 256 --gen-len 32 --percent 0 100 0 100 0 100 --gpu-batch-size 128 --num-gpu-batches 8 --debug fewer_batch --compress-cache"),
    # seq_len = 512, gen_len = 32
    Case("--model facebook/opt-30b --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size 64 --num-gpu-batches 8 --debug fewer_batch --compress-cache"),
    # Case("--model facebook/opt-30b --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size 16 --num-gpu-batches 20 --debug fewer_batch --compress-cache"),
    # seq_len = 1024, gen_len = 32
    Case("--model facebook/opt-30b --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size 20 --num-gpu-batches 12 --debug fewer_batch --compress-cache --prompt-len 1024"),
]

suite_175b_1x1 = [
    # seq_len = 256
    # 1.36 token/s
    Case("--model facebook/opt-175b --path _DUMMY_ --prompt-len 256 --gen-len 32 --pin-weight 0 --percent 0 50 0 0 0 100 --gpu-batch-size 64 --num-gpu-batches 8 --cpu --debug fewer_batch"),
    # seq_len = 512
    Case("--model facebook/opt-175b --path _DUMMY_ --pin-weight 0 --percent 0 50 0 0 0 100 --gpu-batch-size 32 --num-gpu-batches 8 --cpu --debug fewer_batch"),
    # seq_len = 1024
    Case("--model facebook/opt-175b --path _DUMMY_ --pin-weight 0 --percent 0 50 0 0 0 100 --gpu-batch-size 12 --num-gpu-batches 12 --cpu --debug fewer_batch --prompt-len 1024"),
]

suite_175b_1x1_comp = [
    # seq_len = 256
    # 2.26 token/s
    Case("--model facebook/opt-175b --path _DUMMY_ --prompt-len 256 --gen-len 32 --pin-weight 0 --percent 0 100 0 100 0 100 --gpu-batch-size 96 --num-gpu-batches 3 --debug fewer_batch --compress-weight --compress-cache"),
    # seq_len = 512
    Case("--model facebook/opt-175b --path _DUMMY_ --pin-weight 0 --percent 0 100 0 100 0 100 --gpu-batch-size 48 --num-gpu-batches 3 --debug fewer_batch --compress-weight --compress-cache"),
    # seq_len = 1024
    Case("--model facebook/opt-175b --path _DUMMY_ --pin-weight 0 --percent 0 100 0 100 0 100 --gpu-batch-size 12 --num-gpu-batches 4 --debug fewer_batch --compress-weight --compress-cache --prompt-len 1024"),
]

suite_ablation_ds = [
    # 30B
    Case("--model facebook/opt-30b --path _DUMMY_ --percent 0 100 100 0 100 0 --gpu-batch-size 8 --debug fewer_batch"),
    # 175B
    Case("--model facebook/opt-175b --path _DUMMY_ --percent 0 0 100 0 100 0 --gpu-batch-size 2 --debug fewer_batch"),
]

suite_ablation = [
    # 30B

    # 175B
    # no policy search
    Case("--model facebook/opt-175b --path _DUMMY_ --pin-weight 0 --percent 0 50 0 0 0 100 --gpu-batch-size 32 --num-gpu-batches 1 --cpu --debug fewer_batch"),
    # no overlapping
    Case("--model facebook/opt-175b --path _DUMMY_ --pin-weight 0 --percent 0 50 0 0 0 100 --gpu-batch-size 32 --num-gpu-batches 8 --cpu --debug fewer_batch --overlap False"),
    # no cpu compute
    Case("--model facebook/opt-175b --path _DUMMY_ --pin-weight 0 --percent 0 50 0 0 0 100 --gpu-batch-size 32 --num-gpu-batches 8 --debug fewer_batch"),
    # use deepspeed policy
    Case("--model facebook/opt-175b --path _DUMMY_ --percent 0 0 100 0 100 0 --gpu-batch-size 2 --debug fewer_batch"),
]

suite_ablation_policy = [
    # 30B
    Case("--model facebook/opt-30b --path _DUMMY_ --percent 20 80 0 100 0 100 --gpu-batch-size 48 --num-gpu-batches 3 --cpu --debug fewer_batch"),
    Case("--model facebook/opt-30b --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size 48 --num-gpu-batches 3 --cpu --debug fewer_batch"),
    Case("--model facebook/opt-30b --path _DUMMY_ --percent 20 80 0 100 0 100 --gpu-batch-size 48 --num-gpu-batches 1 --cpu --debug fewer_batch"),
    Case("--model facebook/opt-30b --path _DUMMY_ --percent 0 50 0 0 0 100 --gpu-batch-size 32 --num-gpu-batches 8 --cpu --debug fewer_batch", use_page_maga=True),
    Case("--model facebook/opt-30b --path _DUMMY_ --percent 0 0 0 0 0 100 --gpu-batch-size 32 --num-gpu-batches 8 --cpu --debug fewer_batch", use_page_maga=True),
    Case("--model facebook/opt-30b --path _DUMMY_ --percent 0 50 0 0 0 100 --gpu-batch-size 32 --num-gpu-batches 1 --cpu --debug fewer_batch", use_page_maga=True),
    Case("--model facebook/opt-30b --path _DUMMY_ --percent 20 80 100 0 100 0 --gpu-batch-size 1 --num-gpu-batches 1 --cpu --debug fewer_batch"),
    Case("--model facebook/opt-30b --path _DUMMY_ --percent 0 50 100 0 100 0 --gpu-batch-size 1 --num-gpu-batches 1 --cpu --debug fewer_batch", use_page_maga=True),

    # 175B
    Case("--model facebook/opt-175b --path _DUMMY_ --pin-weight 0 --percent 20 80 0 100 0 100 --gpu-batch-size 48 --num-gpu-batches 3 --cpu --debug fewer_batch"),
    Case("--model facebook/opt-175b --path _DUMMY_ --pin-weight 0 --percent 0 100 0 100 0 100 --gpu-batch-size 48 --num-gpu-batches 3 --cpu --debug fewer_batch"),
    Case("--model facebook/opt-175b --path _DUMMY_ --pin-weight 0 --percent 20 80 0 100 0 100 --gpu-batch-size 48 --num-gpu-batches 1 --cpu --debug fewer_batch"),
    Case("--model facebook/opt-175b --path _DUMMY_ --pin-weight 0 --percent 0 50 0 0 0 100 --gpu-batch-size 32 --num-gpu-batches 8 --cpu --debug fewer_batch", use_page_maga=True),
    Case("--model facebook/opt-175b --path _DUMMY_ --pin-weight 0 --percent 0 0 0 0 0 100 --gpu-batch-size 32 --num-gpu-batches 8 --cpu --debug fewer_batch", use_page_maga=True),
    Case("--model facebook/opt-175b --path _DUMMY_ --pin-weight 0 --percent 0 50 0 0 0 100 --gpu-batch-size 32 --num-gpu-batches 1 --cpu --debug fewer_batch", use_page_maga=True),
    Case("--model facebook/opt-175b --path _DUMMY_ --pin-weight 0 --percent 20 80 100 0 100 0 --gpu-batch-size 1 --num-gpu-batches 1 --cpu --debug fewer_batch"),
    Case("--model facebook/opt-175b --path _DUMMY_ --pin-weight 0 --percent 0 50 100 0 100 0 --gpu-batch-size 1 --num-gpu-batches 1 --cpu --debug fewer_batch", use_page_maga=True),
]

suite_175b_breakdown = [
    # seq_len = 512
    Case("--model facebook/opt-175b --path _DUMMY_ --pin-weight 0 --percent 0 50 0 0 0 100 --gpu-batch-size 32 --num-gpu-batches 8 --cpu --debug breakdown"),
]

suite_175b_stage = [
    # 1x1 policy
    Case("--model facebook/opt-175b-stage --path _DUMMY_ --pin-weight 0 --percent 0 50 0 0 0 100 --gpu-batch-size 32 --num-gpu-batches 8 --cpu --debug fewer_batch", "", True),

    # full cpu policy
    Case("--model facebook/opt-175b-stage --path _DUMMY_ --pin-weight 0 --percent 0 100 0 100 0 100 --gpu-batch-size 32 --num-gpu-batches 6 --cpu --debug fewer_batch", "", True),
]

suites = {
    "1b3_test": suite_1b3_test,

    "6b7_1x1": suite_6b7_1x1,
    "6b7_1x1_comp": suite_6b7_1x1_comp,

    "30b_1x1": suite_30b_1x1,
    "30b_1x1_comp": suite_30b_1x1_comp,

    "175b_1x1": suite_175b_1x1,
    "175b_1x1_comp": suite_175b_1x1_comp,

    "ablation": suite_ablation,
    "ablation_policy": suite_ablation_policy,
    "175b_breakdown": suite_175b_breakdown,
    "175b_stage": suite_175b_stage,

    "all_1x1": (suite_6b7_1x1 + suite_6b7_1x1_comp +
                suite_30b_1x1 + suite_30b_1x1_comp +
                suite_175b_1x1 + suite_175b_1x1_comp),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=str, nargs="+")
    parser.add_argument("--log-file", type=str)
    args = parser.parse_args()

    log_file = args.log_file

    for suite in args.suite:
        cases = suites[suite]
        for case in cases:
            config, name, use_page_maga = case.command, case.name, case.use_page_maga
            cmd = f"python -m flexgen.flex_opt {config}"
            if log_file:
                cmd += f" --log-file {args.log_file}"
            if use_page_maga:
                cmd = "bash /usr/local/bin/pagecache-management.sh " + cmd

            if log_file:
                with open(log_file, "a") as f: f.write(f"#### {name}\n```\n{cmd}\n")
            run_cmd(cmd)
            if log_file:
                with open(log_file, "a") as f: f.write(f"```\n")
