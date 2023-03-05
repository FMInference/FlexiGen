"""
Usage:
    python3 helm_real_run.py --model facebook/opt-30b --num-workers 10 --worker-id 0
"""

import argparse
import dataclasses
from typing import list


@dataclasses.dataclass
class SubscenarioConfig:
    description: str
    num_eval: int
    prompt_len: int
    output_len: int


class Policy:
    percent: List[int]
    gpu_batch_size: int
    num_gpu_batches: int


all_scenarios = [
    # 431 scenarios
]


def run_cmd(cmd):
    print(cmd)
    return os.system(cmd)


def get_policy(sce):
    policy_table = [
        (256, 8) : Policy([20, 80, 0, 100, 0, 100], 96, 3),
        (512, 1) : Policy([20, 80, 0, 100, 0, 100], 48, 3),
        (256, 50) : Policy([20, 80, 0, 100, 0, 100], 36, 4),
        (512, 20) : Policy([20, 80, 0, 100, 0, 100], 36, 4),
        (1984, 64) : Policy([0, 100, 0, 100, 0, 100], 8, 4),
    ]

    # round to a policy in polic_table



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--worker-id", type=int)
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    args = parser.parse_args()

    scenarios = all_scenarios[args.worker_id::args.num_workers]

    for sce in scenarios:
        policy = get_policy(sce)

        percent = " ".join([str(policy.percent[i]) for i in range(6)])

        cmd = (f"python3 helm_run.py --description {sce.description} "
               f"--model {args.model} "
               f"--percent {percent} "
               f"--gpu-batch-size {policy.gpu_batch_size} "
               f"--num-gpu-batches {policy.num_gpu_batches} "
               f"--cpu "
               f"--max-eval-instance {policy.num_eval} ")
