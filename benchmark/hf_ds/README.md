# Benchmark Baselines

## Install
Install the forks of Huggingface/transformers and Microsoft/DeepSpeed following this [guide](../third_party/README.md).

```
pip3 install accelerate==0.15.0
```
Install dependencies:
```
sudo apt-get install libaio-dev
```

## Run one case

### HuggingFace Accelerate
```
python3 hf_opt.py --model facebook/opt-1.3b --batch-size 16
```

### DeepSpeed 
```
deepspeed --num_gpus 1 hf_opt.py --model facebook/opt-1.3b --batch-size 16
```

## Run multiple cases
```
python3 bench_hf.py 6b7
python3 bench_hf.py 30b
python3 bench_hf.py 175b
```
