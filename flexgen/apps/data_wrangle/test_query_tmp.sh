python3  ./data_wrangle_run.py\
    --num_run 200 \
    --num_trials 1 \
    --nan_tok "" \
    --do_test \
    --sample_method manual \
    --data_dir data/datasets/entity_matching/structured/DBLP-GoogleScholar \
    --batch_run --pad-to-seq-len 1209 --model facebook/opt-30b --percent 10 90 0 100 0 100 --gpu-batch-size 10 --num-gpu-batches 5