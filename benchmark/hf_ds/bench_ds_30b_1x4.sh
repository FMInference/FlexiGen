deepspeed --num_gpus 4 hf_opt.py --model facebook/opt-30b --batch-size 24 --cut-gen-len 5 --cpu --dummy
