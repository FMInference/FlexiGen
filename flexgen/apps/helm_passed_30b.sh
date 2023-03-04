model=facebook/opt-iml-30b

# WikiFact (plaintiff), 10m
time python3 helm_run.py --description wikifact:model=text,k=5,subject=plaintiff --model $model --percent 20 80 0 100 0 100 --gpu-batch-size 96 --num-gpu-batches 3 --cpu --max-eval-instance 96  # 96

# WikiFact (instance_of), 55m
time python3 helm_run.py --description wikifact:model=text,k=5,subject=instance_of --model $model --percent 20 80 0 100 0 100 --gpu-batch-size 96 --num-gpu-batches 3 --cpu --max-eval-instance 850  # 850

# MMLU (abstract_algebra), 31m
time python3 helm_run.py --description mmlu:model=together/opt-175b,subject=abstract_algebra,data_augmentation=canonical --model $model --percent 20 80 0 100 0 100 --gpu-batch-size 48 --num-gpu-batches 3 --cpu --max-eval-instance 100  # 100

# MMLU (us_foreign_policy), 33m
time python3 helm_run.py --description mmlu:model=together/opt-175b,subject=us_foreign_policy,data_augmentation=canonical --model $model --percent 20 80 0 100 0 100 --gpu-batch-size 48 --num-gpu-batches 3 --cpu --max-eval-instance 100  # 100

# Synthetic reasoning (abstract symbols, pattern_match), 118m
time python3 helm_run.py --description synthetic_reasoning:model=together/opt-175b,mode=pattern_match --model $model --percent 20 80 0 100 0 100 --gpu-batch-size 36 --num-gpu-batches 4 --cpu --max-eval-instance 515  # 515

# Synthetic reasoning (natural language), 100m
time python3 helm_run.py --description synthetic_reasoning_natural:model=together/opt-175b,difficulty=easy --model $model --percent 20 80 0 100 0 100 --gpu-batch-size 36 --num-gpu-batches 4 --cpu --max-eval-instance 515  # 515

# XSUM, 902m
time python3 helm_run.py --description summarization_xsum_sampled:model=text,temperature=0.3,device=cpu --model $model --percent 0 100 0 100 0 100 --gpu-batch-size 8 --num-gpu-batches 4 --cpu --max-eval-instance 518  # 518
