model=facebook/opt-30b

# WikiFact (plaintiff), 10m
time python3 helm_run.py --description wikifact:model=text,k=5,subject=plaintiff --model $model --percent 20 80 0 100 0 100 --gpu-batch-size 96 --num-gpu-batches 3 --cpu --max-eval-instance 96  # 96

# WikiFact (instance_of), 60m
time python3 helm_run.py --description wikifact:model=text,k=5,subject=instance_of --model $model --percent 20 80 0 100 0 100 --gpu-batch-size 96 --num-gpu-batches 3 --cpu --max-eval-instance 850  # 850

# MMLU (abstract_algebra), 34m
time python3 helm_run.py --description mmlu:model=together/opt-175b,subject=abstract_algebra,data_augmentation=canonical --model $model --percent 20 80 0 100 0 100 --gpu-batch-size 48 --num-gpu-batches 3 --cpu --max-eval-instance 100  # 100

# MMLU (us_foreign_policy), 35m
time python3 helm_run.py --description mmlu:model=together/opt-175b,subject=us_foreign_policy,data_augmentation=canonical --model $model --percent 20 80 0 100 0 100 --gpu-batch-size 48 --num-gpu-batches 3 --cpu --max-eval-instance 100  # 100

# Synthetic reasoning (abstract symbols), 113m
time python3 helm_run.py --description synthetic_reasoning_natural:model=together/opt-175b,difficulty=easy --model $model --percent 20 80 0 100 0 100 --gpu-batch-size 36 --num-gpu-batches 4 --cpu --max-eval-instance 515  # 515

# Synthetic reasoning (natural language), 114m
time python3 helm_run.py --description synthetic_reasoning_natural:model=together/opt-175b,difficulty=easy --model $model --percent 20 80 0 100 0 100 --gpu-batch-size 36 --num-gpu-batches 4 --cpu --max-eval-instance 515  # 515
