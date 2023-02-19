# Benchmark FlexGen
NOTE: The benchmark is done with dummy weights for faster experiments.

## Single GPU

### OPT-6.7B
```
# fp16
python3 bench_suite.py 6b7_1x1

# with int4 compression
python3 bench_suite.py 6b7_1x1_comp
```

### OPT-30B
```
# fp16
python3 bench_suite.py 30b_1x1

# with int4 compression
python3 bench_suite.py 30b_1x1_comp
```

### OPT-175B
```
# fp16
python3 bench_suite.py 175b_1x1

# with int4 compression
python3 bench_suite.py 175b_1x1_comp
```

## Distributed GPUs

### OPT-6.7B
```
# 1 node with 4 GPUs
bash bench_6.7b_1x4.sh

# 4 nodes and one GPU per node
bash bench_6.7b_4x1.sh
```

### OPT-30B
```
# 1 node with 4 GPUs
bash bench_30b_1x4.sh

# 4 nodes and one GPU per node
bash bench_30b_4x1.sh
```

### OPT-175B
```
# 1 node with 4 GPUs
bash bench_175b_1x4.sh

# 4 nodes and one GPU per node
bash bench_175b_4x1.sh
```
