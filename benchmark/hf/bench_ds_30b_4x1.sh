deepspeed --num_nodes 4 --num_gpus 1 --master_port 7778 --hostfile hostfile \
    hf_opt.py --model facebook/opt-30b --batch-size 24 --cut-gen-len 5 --dummy --cpu
