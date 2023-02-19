deepspeed --num_nodes 4 --num_gpus 1 --master_port 7778 --hostfile hostfile \
    hf_opt.py --model facebook/opt-175b --batch-size 4 --cut-gen-len 5 --dummy --cpu
