## Example commands

### Completion
Example commands for OPT-30B and OPT-66B on machines with 32GB of system RAM and 24 GB of VRAM.
```
python completion.py --model facebook/opt-30b --percent 100 0 100 0 100 0 --compress-weight
python completion.py --model facebook/opt-66b --percent 50 10 100 0 100 0 --compress-weight
```

### Data Wrangling

Run the tests of data wrangling tasks in the [fm_data_tasks](https://github.com/HazyResearch/fm_data_tasks) repo from [HazyResearch](https://github.com/HazyResearch). 
Check [more details](./data_wrangle/README.md).
```
cd data_wrangle
bash install
bash test_batch_query_all_opt6.7b.sh
bash test_batch_query_all_opt30b.sh
bash test_batch_query_all_opt175b.sh
```


### HELM benchmark
Run Massive Multitask Language Understanding (MMLU) scenario.
```
python3 helm_run.py --description mmlu:model=text,subject=abstract_algebra,data_augmentation=canonical --pad-to-seq-len 512 --model facebook/opt-30b --percent 20 80 0 100 0 100 --gpu-batch-size 48 --num-gpu-batches 3 --max-eval-instance 100
```
