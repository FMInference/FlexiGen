model=facebook/opt-30b

# WikiFact (plaintiff)
time python3 helm_run.py --description wikifact:model=text,k=5,subject=plaintiff --model $model --percent 20 80 0 100 0 100 --gpu-batch-size 96 --num-gpu-batches 3 --cpu --max-eval-instance 10  # 96

# WikiFact (instance_of)
time python3 helm_run.py --description wikifact:model=text,k=5,subject=instance_of --model $model --percent 20 80 0 100 0 100 --gpu-batch-size 96 --num-gpu-batches 3 --cpu --max-eval-instance 10 # 850

# MMLU (abstract_algebra)
time python3 helm_run.py --description mmlu:model=together/opt-175b,subject=abstract_algebra,data_augmentation=canonical --model $model --percent 20 80 0 100 0 100 --gpu-batch-size 48 --num-gpu-batches 3 --cpu --max-eval-instance 10  # 100

# MMLU (us_foreign_policy)
time python3 helm_run.py --description mmlu:model=together/opt-175b,subject=us_foreign_policy,data_augmentation=canonical --model $model --percent 20 80 0 100 0 100 --gpu-batch-size 48 --num-gpu-batches 3 --cpu --max-eval-instance 10  # 100

# Synthetic reasoning (abstract symbols)
time python3 helm_run.py --description synthetic_reasoning_natural:model=together/opt-175b,difficulty=easy --model $model --percent 20 80 0 100 0 100 --gpu-batch-size 36 --num-gpu-batches 4 --cpu --max-eval-instance 10  # 515

# Synthetic reasoning (natural language)
time python3 helm_run.py --description synthetic_reasoning_natural:model=together/opt-175b,difficulty=easy --model $model --percent 20 80 0 100 0 100 --gpu-batch-size 36 --num-gpu-batches 4 --cpu --max-eval-instance 10  # 515
