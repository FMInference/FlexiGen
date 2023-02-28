python3 hf_opt.py --num-gpus 4 --model facebook/opt-6.7b --dummy --cut-gen-len 5 --batch-size 16
deepspeed --num_gpus 4 hf_opt.py --model facebook/opt-6.7b --dummy --cut-gen-len 5 --batch-size 48

python3 hf_opt.py --num-gpus 4 --model facebook/opt-30b  --dummy --cut-gen-len 5 --batch-size 8 --cpu
deepspeed --num_gpus 4 hf_opt.py --model facebook/opt-30b  --dummy --cut-gen-len 5 --batch-size 24 --cpu

python3 hf_opt.py --num-gpus 4 --model facebook/opt-175b --dummy --cut-gen-len 5 --batch-size 2 --cpu
deepspeed --num_gpus 4 hf_opt.py --model facebook/opt-175b --dummy --cut-gen-len 5 --batch-size 4 --cpu
