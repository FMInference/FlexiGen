python3  ./data_wrangle_run.py\
    --num_trials 1 \
    --nan_tok "" \
    --do_test \
    --sample_method manual \
    --data_dir data/datasets/entity_matching/structured/Fodors-Zagats \
    --pad-to-seq-len 32 --model facebook/opt-6.7b --percent 100 0 100 0 100 0 --gpu-batch-size 1 --num-gpu-batches 1

python3  ./data_wrangle_run.py\
    --num_trials 1 \
    --nan_tok "" \
    --do_test \
    --sample_method manual \
    --data_dir data/datasets/entity_matching/structured/Beer \
    --pad-to-seq-len 32 --model facebook/opt-6.7b --percent 100 0 100 0 100 0 --gpu-batch-size 1 --num-gpu-batches 1

python3  ./data_wrangle_run.py\
    --num_trials 1 \
    --nan_tok "" \
    --do_test \
    --sample_method manual \
    --data_dir data/datasets/entity_matching/structured/iTunes-Amazon \
    --pad-to-seq-len 32 --model facebook/opt-6.7b --percent 100 0 100 0 100 0 --gpu-batch-size 1 --num-gpu-batches 1

python3  ./data_wrangle_run.py\
    --num_trials 1 \
    --nan_tok "" \
    --do_test \
    --sample_method manual \
    --data_dir data/datasets/entity_matching/structured/Walmart-Amazon \
    --pad-to-seq-len 32 --model facebook/opt-6.7b --percent 100 0 100 0 100 0 --gpu-batch-size 1 --num-gpu-batches 1

python3  ./data_wrangle_run.py\
    --num_trials 1 \
    --nan_tok "" \
    --do_test \
    --sample_method manual \
    --data_dir data/datasets/entity_matching/structured/Amazon-Google \
    --pad-to-seq-len 32 --model facebook/opt-6.7b --percent 100 0 100 0 100 0 --gpu-batch-size 1 --num-gpu-batches 1

python3  ./data_wrangle_run.py\
    --num_trials 1 \
    --nan_tok "" \
    --do_test \
    --sample_method manual \
    --data_dir data/datasets/entity_matching/structured/DBLP-ACM \
    --pad-to-seq-len 32 --model facebook/opt-6.7b --percent 100 0 100 0 100 0 --gpu-batch-size 1 --num-gpu-batches 1

python3  ./data_wrangle_run.py\
    --num_trials 1 \
    --nan_tok "" \
    --do_test \
    --sample_method manual \
    --data_dir data/datasets/entity_matching/structured/DBLP-GoogleScholar \
    --pad-to-seq-len 32 --model facebook/opt-6.7b --percent 100 0 100 0 100 0 --gpu-batch-size 1 --num-gpu-batches 1

python3  ./data_wrangle_run.py\
    --num_trials 1 \
    --max_tokens 5 \
    --do_test \
    --sample_method manual \
    --data_dir data/datasets/data_imputation/Restaurant \
    --pad-to-seq-len 32 --model facebook/opt-6.7b --percent 100 0 100 0 100 0 --gpu-batch-size 1 --num-gpu-batches 1

python3  ./data_wrangle_run.py\
    --num_trials 1 \
    --max_tokens 10 \
    --do_test \
    --sample_method manual \
    --data_dir data/datasets/data_imputation/Buy \
    --pad-to-seq-len 32 --model facebook/opt-6.7b --percent 100 0 100 0 100 0 --gpu-batch-size 1 --num-gpu-batches 1

python3  ./data_wrangle_run.py\
    --num_trials 1 \
    --do_test \
    --sample_method manual \
    --data_dir data/datasets/error_detection/Hospital \
    --pad-to-seq-len 32 --model facebook/opt-6.7b --percent 100 0 100 0 100 0 --gpu-batch-size 1 --num-gpu-batches 1