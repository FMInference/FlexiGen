import os

def run_cmd(cmd):
    print(cmd)
    return os.system(cmd)


if __name__ == "__main__":
    passed = [
        "boolq:model=text,data_augmentation=canonical",
        "narrative_qa:model=text,data_augmentation=canonical",
        # "news_qa:model=text,data_augmentation=canonical"  # need to download the dataset manually.
        "quac:model=text,data_augmentation=canonical",
        # 2
        "natural_qa:model=text,mode=openbook_longans,data_augmentation=canonical",
        # 3
        "commonsense:model=text,dataset=commonsenseqa,method=multiple_choice_separate_calibrated,data_augmentation=canonical",
        "truthful_qa:model=text,task=mc_single,data_augmentation=canonical",
        # 57
        "mmlu:model=text,subject=abstract_algebra,data_augmentation=canonical",
        "summarization_cnndm:model=text,temperature=0.3,device=cpu",
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
        "babi_qa:model=text_code,task=1",
        # 3
        "dyck_language:model=text_code,num_parenthesis_pairs=2",
        # 70
        "math:model=text_code,subject=number_theory,level=1,use_official_examples=True",
        "gsm:model=text_code",
        # 5
        "lsat_qa:model=text_code,task=all",
        # 18
        "lextreme:subset=brazilian_court_decisions_judgment,model=all",
        # 7
        "lex_glue:subset=ecthr_a,model=all",
        # 3
        "entity_matching:model=text,dataset=Beer",
        # 2
        "entity_data_imputation:model=text,dataset=Buy",
        # 12
        "bbq:model=text,subject=all",
        # 6
        "bold:model=text,subject=all",
        "boolq:model=text,only_contrast=True,data_augmentation=contrast_sets",
    ]

    descriptions = [
        # 2 unexpected keyword arguemtn 'track'
        # "msmarco:model=full_functionality_text,data_augmentation=canonical,track=regular,valid_topk=30",
        # data retrieval failed with wget
        # "imdb:model=text,data_augmentation=canonical",
        # 12 data retrieval failed with wget
        # "blimp:model=full_functionality_text,phenomenon=anaphor_agreement",
        # data retrieval failed with wget
        # "wikitext_103:model=full_functionality_text",
        # 22 data retrieval failed with wget
        # "the_pile:model=full_functionality_text,subset=ArXiv",
        # 2 data retrival failed with wget
        # "twitter_aae:model=full_functionality_text,demographic=aa",
        # 42 need to download the dataset manually
        # "ice:model=full_functionality_text,subset=can",
        # 4 scenario_state.request_states is empty
        # "numeracy:model=text_code,run_solver=True,relation_type=linear,mode=function",
        # data retrival failed with wget
        # "legal_support:model=text_code",
        # download failed gdown
        # "med_qa:model=biomedical",
        # 2 connection failed
        # "code:model=code,dataset=humaneval",
        # 5 download failed gdown
        # "copyright:model=text,datatag=n_books_1000-extractions_per_book_1-prefix_length_125",
        # 3 download failed gdown
        # "disinformation:model=text,capability=reiteration,topic=climate",
        # data retrival failed with wget
        # "real_toxicity_prompts:model=text",
        # 2 data retrival failed with wget
        # "synthetic_efficiency:model=text,tokenizer=default,num_prompt_tokens=default_sweep,num_output_tokens=default_sweep",
        # data retrival failed with wget
        # "imdb:model=text,only_contrast=True,data_augmentation=contrast_sets",
    ]

    torun = descriptions
    for i, des in enumerate(torun):
        print("=" * 10 + f" {i+1}/{len(torun)} : {des} " + "=" * 10)
        des = des.replace("model=text_code", "model=together/opt-175b")
        des = des.replace("model=text", "model=together/opt-175b")
        des = des.replace("model=all", "model=together/opt-175b")
        cmd = (f"python3 helm_run.py --description {des} "
               f"--model facebook/opt-125m --percent 100 0 100 0 100 0 --gpu-batch-size 32 "
               f"--num-gpu-batches 1 --max-eval-instance 10")
        ret = run_cmd(cmd)
        assert ret == 0
