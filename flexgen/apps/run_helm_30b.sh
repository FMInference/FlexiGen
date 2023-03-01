model=facebook/opt-30b

# MMLU
time python3 helm_run.py --description mmlu:model=text,subject=abstract_algebra,data_augmentation=canonical --pad-to-seq-len 512 --model $model --percent 20 80 0 100 0 100 --gpu-batch-size 48 --num-gpu-batches 3 --cpu --max-eval-instance 100

# Synthetic reasoning
time python3 helm_run.py --description synthetic_reasoning_natural:model=text_code,difficulty=easy --pad-to-seq-len 512 --model $model --percent 20 80 0 100 0 100 --gpu-batch-size 36 --num-gpu-batches 4 --cpu --max-eval-instance 515

# XSUM
#time python3 helm_run.py --description summarization_xsum_sampled:model=text,temperature=0.3 --model $model --percent 20 80 0 100 0 100 --gpu-batch-size 8 --num-gpu-batches 4 --cpu --max-eval-instance 100
