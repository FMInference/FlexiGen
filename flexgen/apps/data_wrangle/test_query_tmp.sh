python3  ./data_wrangle_run.py\
    --num_run 109 \
    --num_trials 1 \
    --nan_tok "" \
    --do_test \
    --sample_method manual \
    --data_dir data/datasets/entity_matching/structured/iTunes-Amazon \
    --batch_run --pad-to-seq-len 529 --model facebook/opt-175b --pin-weight 0 --cpu --percent 0 50 0 0 0 100 --gpu-batch-size 20 --num-gpu-batches 5
