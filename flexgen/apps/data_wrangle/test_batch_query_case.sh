python3  ./data_wrangle_run.py\
    --num_run 64 \
    --num_trials 1 \
    --do_test \
    --sample_method manual \
    --data_dir data/datasets/data_imputation/Restaurant \
    --batch_run \
    --pad-to-seq-len 128 --model facebook/opt-6.7b --percent 100 0 100 0 100 0 --gpu-batch-size 8 --num-gpu-batches 8