model=facebook/opt-30b

# HellaSwag
time python3 helm_run.py --description commonsense:model=together/opt-175b,dataset=hellaswag,method=multiple_choice_separate_original,data_augmentation=canonical --model $model --percent 20 80 0 100 0 100 --gpu-batch-size 96 --num-gpu-batches 3 --cpu --max-eval-instance 1000

# DYCK
time python3 helm_run.py --description dyck_language:model=together/opt-175b,num_parenthesis_pairs=2 --model $model --percent 20 80 0 100 0 100 --gpu-batch-size 96 --num-gpu-batches 3 --cpu --max-eval-instance 500

# MMLU
time python3 helm_run.py --description mmlu:model=together/opt-175b,subject=abstract_algebra,data_augmentation=canonical --model $model --percent 20 80 0 100 0 100 --gpu-batch-size 48 --num-gpu-batches 3 --cpu --max-eval-instance 100

# MMLU
time python3 helm_run.py --description mmlu:model=together/opt-175b,subject=college_chemistry,data_augmentation=canonical --model $model --percent 20 80 0 100 0 100 --gpu-batch-size 48 --num-gpu-batches 3 --cpu --max-eval-instance 100

# Synthetic reasoning
time python3 helm_run.py --description synthetic_reasoning_natural:model=together/opt-175b,difficulty=easy --model $model --percent 20 80 0 100 0 100 --gpu-batch-size 36 --num-gpu-batches 4 --cpu --max-eval-instance 515
