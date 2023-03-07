import argparse
from dataclasses import dataclass

from flexgen.utils import run_cmd


@dataclass
class Case:
    command: str
    name: str = ""
    use_page_maga: bool = False


suite_175b = [
    # seq_len = 32, gen_len = 256
    # Case("--model facebook/opt-175b --path _DUMMY_ --prompt-len 32 --gen-len 8 --pin-weight 0 --percent 0 50 0 0 0 100 --gpu-batch-size 64 --num-gpu-batches 8 --cpu --debug fewer_batch"),
    Case("--model facebook/opt-175b --path _DUMMY_ --prompt-len 280 --gen-len 8 --pin-weight 0 --percent 0 50 0 0 0 100 --gpu-batch-size 64 --num-gpu-batches 8 --cpu --debug fewer_batch"),
]

suite_175b_comp = [
    # seq_len = 32, gen_len = 256
]

suites = {
    "175b_1x1_32_256": suite_175b_1x1,
    "175b_1x1_comp_32_256": suite_175b_1x1_comp,

    "all_1x1": (suite_6b7_1x1_32_256 + suite_6b7_1x1_comp +
                suite_30b_1x1_32_256 + suite_30b_1x1_comp +
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
