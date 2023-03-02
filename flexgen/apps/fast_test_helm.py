import os

def run_cmd(cmd):
    print(cmd)
    return os.system(cmd)


if __name__ == "__main__":
    passed = [
        # 57
        "mmlu:model=text,subject=abstract_algebra,data_augmentation=canonical",
        # 1
        "summarization_cnndm:model=text,temperature=0.3,device=cpu",
        # 1
        "summarization_xsum_sampled:model=text,temperature=0.3,device=cpu",
        # 11
        "raft:subset=ade_corpus_v2,model=text,data_augmentation=canonical",
        # 9
        "civil_comments:model=text,demographic=all,data_augmentation=canonical",
        # 86
        "wikifact:model=text,k=5,subject=plaintiff",
        # 3
        "synthetic_reasoning:model=text_code,mode=pattern_match",
        # 2
        "synthetic_reasoning_natural:model=text_code,difficulty=easy",
        # 21
        "babi_qa:model=text_code,task=1"
        # 3
        "dyck_language:model=text_code,num_parenthesis_pairs=2"
        # 70
        "math:model=text_code,subject=number_theory,level=1,use_official_examples=True",
        # 1
        "gsm:model=text_code",
        # 1
        "legal_support:model=text_code",
    ]

    descriptions = [
    ]

    torun = passed
    for i, des in enumerate(torun):
        print("=" * 10 + f" {i+1}/{len(torun)} : {des} " + "=" * 10)
        cmd = (f"python3 helm_run.py --description {des} "
               f"--model facebook/opt-125m --percent 100 0 100 0 100 0 --gpu-batch-size 32 "
               f"--num-gpu-batches 1 --max-eval-instance 10")
        ret = run_cmd(cmd)
        assert ret == 0
