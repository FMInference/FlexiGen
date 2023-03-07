import os

def run_cmd(cmd):
    print(cmd)
    return os.system(cmd)


if __name__ == "__main__":
    passed = [
        "boolq:model=text,data_augmentation=canonical",
        "narrative_qa:model=text,data_augmentation=canonical",
        "quac:model=text,data_augmentation=canonical",
        # 2
        "natural_qa:model=text,mode=openbook_longans,data_augmentation=canonical",
        # 3
        "commonsense:model=text,dataset=commonsenseqa,method=multiple_choice_separate_calibrated,data_augmentation=canonical",
        "truthful_qa:model=text,task=mc_single,data_augmentation=canonical",
        # 57
        "mmlu:model=text,subject=abstract_algebra,data_augmentation=canonical",
        # 2
        "msmarco:model=full_functionality_text,data_augmentation=canonical,track=regular,valid_topk=30",
        "summarization_cnndm:model=text,temperature=0.3,device=cpu",
        "summarization_xsum_sampled:model=text,temperature=0.3,device=cpu",
        "imdb:model=text,data_augmentation=canonical",
        # 11
        "raft:subset=ade_corpus_v2,model=text,data_augmentation=canonical",
        # 9
        "civil_comments:model=text,demographic=all,data_augmentation=canonical",
        # 12
        "blimp:model=full_functionality_text,phenomenon=anaphor_agreement",
        "wikitext_103:model=full_functionality_text",
        # 2
        "twitter_aae:model=full_functionality_text,demographic=aa",
        # 86
        "wikifact:model=text,k=5,subject=plaintiff",
        # 3
        "synthetic_reasoning:model=text_code,mode=pattern_match",
        # 2
        "synthetic_reasoning_natural:model=text_code,difficulty=easy",
        # 21
        "babi_qa:model=text_code,task=1",
        # 3
        "dyck_language:model=text_code,num_parenthesis_pairs=2",
        # 70
        "math:model=text_code,subject=number_theory,level=1,use_official_examples=True",
        "gsm:model=text_code",
        "legal_support:model=text_code",
        # 5
        "lsat_qa:model=text_code,task=all",
        # 18
        "lextreme:subset=brazilian_court_decisions_judgment,model=all",
        # 7
        "lex_glue:subset=ecthr_a,model=all",
        "med_qa:model=biomedical",
        # 3
        "entity_matching:model=text,dataset=Beer",
        # 2
        "entity_data_imputation:model=text,dataset=Buy",
        # 5
        "copyright:model=text,datatag=n_books_1000-extractions_per_book_1-prefix_length_125",
        # 3
        "disinformation:model=text,capability=reiteration,topic=climate",
        # 12
        "bbq:model=text,subject=all",
        "real_toxicity_prompts:model=text",
        # 6
        "bold:model=text,subject=all",
        # 2
        "synthetic_efficiency:model=text,tokenizer=default,num_prompt_tokens=default_sweep,num_output_tokens=default_sweep",
        "boolq:model=text,only_contrast=True,data_augmentation=contrast_sets",
        "imdb:model=text,only_contrast=True,data_augmentation=contrast_sets",
    ]

    descriptions = [
        # 1 need to download the dataset manually.
        # "news_qa:model=text,data_augmentation=canonical"
        # 22 data retrieval failed with wget
        # "the_pile:model=full_functionality_text,subset=ArXiv",
        # 42 need to download the dataset manually
        # "ice:model=full_functionality_text,subset=can",
        # 4 cannot run due to a recent bug/refactor in HELM
        # "numeracy:model=text_code,run_solver=True,relation_type=linear,mode=function",
        # 2 Only support code model
        # "code:model=code,dataset=humaneval",
    ]

    torun = passed
    for i, des in enumerate(torun):
        print("=" * 10 + f" {i+1}/{len(torun)} : {des} " + "=" * 10)
        for name in ["model=text_code","model=text","model=code","model=all","model=full_functionality_text"]:
            des = des.replace(name, "model=together/opt-175b")
        cmd = (f"python3 helm_run.py --description {des} "
               f"--model facebook/opt-125m --percent 100 0 100 0 100 0 --gpu-batch-size 32 "
               f"--num-gpu-batches 1 --max-eval-instance 10")
        ret = run_cmd(cmd)
        assert ret == 0
