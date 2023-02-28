deepspeed --num_gpus 4 hf_opt.py --model facebook/opt-6.7b --batch-size 48 --cut-gen-len 5 --dummy
