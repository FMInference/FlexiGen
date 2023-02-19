"""
Usage:
bash /usr/local/bin/pagecache-management.sh python3 profile_bandwidth.py
"""

import argparse
import numpy as np
import os
import time
import torch

from flexgen.utils import GB, MB, KB


def benchmark_func(func, number, repeat, warmup=3):
    for i in range(warmup):
        func()

    costs = [0]

    for i in range(repeat):
        torch.cuda.synchronize()
        tic = time.time()
        for i in range(number):
            func()
        torch.cuda.synchronize()
        costs.append((time.time() - tic) / number)

    return costs


def profile_bandwidth(path):
    s, h = 512, 512
    path_dir = os.path.dirname(path)
    os.makedirs(path_dir, exist_ok=True)

    links = [("cpu", "gpu"), ("gpu", "cpu"), ("gpu", "gpu"), ("cpu", "cpu"),
             ("cpu", "disk"), ("disk", "cpu")]

    for (dst, src) in links:
        for b in [1, 128, 512]:
            if dst == "cpu":
                dst_tensor = torch.ones((b, s, h), dtype=torch.int8, pin_memory=True)
            elif dst == "gpu":
                dst_tensor = torch.ones((b, s, h), dtype=torch.int8, device="cuda:0")
            elif dst == "disk":
                np.lib.format.open_memmap(path, mode="w+", shape=((b,s,h)), dtype=np.int8)
                dst_tensor = path

            if src == "cpu":
                src_tensor = torch.ones((b, s, h), dtype=torch.int8, pin_memory=True)
            elif src == "gpu":
                src_tensor = torch.ones((b, s, h), dtype=torch.int8, device="cuda:0")
            elif src == "disk":
                np.lib.format.open_memmap(path, mode="w+", shape=((b,s,h)), dtype=np.int8)
                src_tensor = path

            dst_indices = (slice(0, b), slice(0, s), slice(0, h))
            src_indices = (slice(0, b), slice(0, s), slice(0, h))

            def func():
                if isinstance(src_tensor, str):
                    src_tensor_ = torch.from_numpy(np.lib.format.open_memmap(src_tensor))
                else:
                    src_tensor_ = src_tensor
                if isinstance(dst_tensor, str):
                    dst_tensor_ = torch.from_numpy(np.lib.format.open_memmap(dst_tensor))
                else:
                    dst_tensor_ = dst_tensor
                dst_tensor_[dst_indices].copy_(src_tensor_[src_indices])

            size = np.prod([(x.stop - x.start) / (x.step or 1) for x in dst_indices])
            cost = np.mean(benchmark_func(func, number=5, repeat=3))
            bandwidth = size / cost / GB

            print(f"size: {size / MB:6.2f} MB, {src}-to-{dst} bandwidth: {bandwidth:.3f} GB/s")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offload-path", type=str, default="~/flexgen_offload_dir/tmp.npy")
    args = parser.parse_args()

    profile_bandwidth(os.path.expanduser(args.offload_path))
