# Benchmark FlexGen
NOTE: This benchmark uses dummy weights by default for faster experiments.
It is expected if you see randomly generated garbled characters, but the throughput and latency numbers should be correct.

## Mount SSD
The following commands use `~/flexgen_offload_dir` as the offloading folder by default.
To get the best performance, it is recommonded to mount this folder on a fast SSD.
If you use AWS or GCP instances with local SSDs, you can use [mount_nvme_aws.sh](../../scripts/mount_nvme_aws.sh) or [mount_nvme_gcp.sh](../../scripts/mount_nvme_gcp.sh) to mount the local SSDs.

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

### Requirements
```
sudo apt install openmpi-bin
```

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
