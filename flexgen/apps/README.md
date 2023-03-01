## Example commands

### Completion
Example commands for OPT-30B and OPT-66B on machines with 32GB of system RAM and 24 GB of VRAM.
```
python completion.py --model facebook/opt-30b --percent 100 0 100 0 100 0 --compress-weight
python completion.py --model facebook/opt-66b --percent 50 10 100 0 100 0 --compress-weight
```

### HELM benchmark
Run Massive Multitask Language Understanding (MMLU) scenario.
```
python3 helm_run.py --description mmlu:model=text,subject=abstract_algebra,data_augmentation=canonical --pad-to-seq-len 512 --model facebook/opt-30b --percent 20 80 0 100 0 100 --gpu-batch-size 48 --num-gpu-batches 3 --max-eval-instance 100
```
