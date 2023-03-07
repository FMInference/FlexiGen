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
    ScenarioConfig("boolq:model=text,data_augmentation=canonical", 1000, 908, 5),
    ScenarioConfig("narrative_qa:model=text,data_augmentation=canonical", 355, 1652, 51),
    # ScenarioConfig("news_qa", ),
    ScenarioConfig("quac:model=text,data_augmentation=canonical", 1000, 1645, 22),
    ScenarioConfig("natural_qa:model=text,mode=openbook_longans,data_augmentation=canonical", 
                   1000, 1420, 212),
    # ScenarioConfig("natural_qa/closedbook", 1000, 112, 153),
    ScenarioConfig("commonsense:model=text,dataset=commonsenseqa,method=multiple_choice_separate_calibrated,data_augmentation=canonical")
    ScenarioConfig("truthful_qa:model=text,task=mc_single,data_augmentation=canonical",
                   654, 405, 1),
    ScenarioConfig("mmlu:model=text,subject=abstract_algebra,data_augmentation=canonical",
                   102.8, 468, 1),
    ScenarioConfig("msmarco:model=full_functionality_text,data_augmentation=canonical,track=regular,valid_topk=30",
                   1000, 533, 5),
    ScenarioConfig("summarization_cnndm:model=text,temperature=0.3,device=cpu", 466, 1550, 115),
    ScenarioConfig("summarization_xsum_sampled:model=text,temperature=0.3,device=cpu",
                   518, 1510, 24),
    ScenarioConfig("imdb:model=text,data_augmentation=canonical", 1000, 1389, 5),
    ScenarioConfig("raft:subset=ade_corpus_v2,model=text,data_augmentation=canonical",
                   40, 813, 18.7),
    ScenarioConfig("civil_comments:model=text,demographic=all,data_augmentation=canonical",
                   371.6, 722.6, 5),
    ScenarioConfig("blimp:model=full_functionality_text,phenomenon=anaphor_agreement",
                   1000, 8.6, 0),
    ScenarioConfig("wikitext_103", 1)
    # ScenarioConfig("the_pile", 492, 1374, 0),
    ScenarioConfig("twitter_aae:model=full_functionality_text,demographic=aa", 1000, 14.86, 0),
    # ScenarioConfig("ice", 490, 2040, 0),
    ScenarioConfig("wikifact:model=text,k=5,subject=plaintiff", 746, 83, 40),
    ScenarioConfig("synthetic_reasoning:model=text_code,mode=pattern_match", 515, 286, 14),
    ScenarioConfig("synthetic_reasoning_natural:model=text_code,difficulty=easy", 515, 400, 9),
    ScenarioConfig("babi_qa:model=text_code,task=1", 906, 469, 1.2),
    ScenarioConfig("dyck_language:model=text_code,num_parenthesis_pairs=2", 500, 164.4, 5),
    ScenarioConfig("math:model=text_code,subject=number_theory,level=1,use_official_examples=True",
                   62.4, 402, 2.6),
    ScenarioConfig("gsm:model=text_code", 1000, 880, 400),
    ScenarioConfig("legal_support:model=text_code", 489, 605, 1),
    ScenarioConfig("lsat_qa:model=text_code,task=all", 230, 1190, 1),
    ScenarioConfig("lextreme:subset=brazilian_court_decisions_judgment,model=all", 18)
    ScenarioConfig("lex_glue:subset=ecthr_a,model=all", 7)
    ScenarioConfig("med_qa:model=biomedical", 1)
    ScenarioConfig("entity_matching:model=text,dataset=Beer", 231, 890, 5),
    ScenarioConfig("entity_data_imputation:model=text,dataset=Buy", 76, 308, 5),
    # ScenarioConfig("code", 2) no opt data point
    ScenarioConfig("copyright:model=text,datatag=n_books_1000-extractions_per_book_1-prefix_length_125", 375, 118, 4480),
    # ScenarioConfig("copyright_code", 3)
    ScenarioConfig("disinformation:model=text,capability=reiteration,topic=climate""disinformation_reiteration"),
    ScenarioConfig("bbq:model=text,subject=all", 1000, 424, 1),
    ScenarioConfig("real_toxicity_prompts:model=text", 500, 15.5, 100),
    ScenarioConfig("bold:model=text,subject=all", 1000, 11.8, 20),
    ScenarioConfig("synthetic_efficiency:model=text,tokenizer=default,num_prompt_tokens=default_sweep,num_output_tokens=default_sweep", 10, 666, 32),
    ScenarioConfig("boolq:model=text,only_contrast=True,data_augmentation=contrast_sets", 1000, 908, 5),
    ScenarioConfig("imdb:model=text,only_contrast=True,data_augmentation=contrast_sets", 1000, 1553, 1),
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
