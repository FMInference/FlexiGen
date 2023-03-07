## Effective Batch Size of Each System

### Setup
- Hardware: an NVIDIA T4 (16GB) instance on GCP with 208GB of DRAM and 1.5TB of SSD.  
- Workload: input sequence length = 512, output sequence length = 32.

### Effective Batch Size

The table below lists the effective batch size of each system.
The device in the bracket denotes the lowest level of memory hierarchy that the system needs for offloading.
The batch size is tuned for each system to achieve its maximum throughput with the following principle:
- Find a level of memory hierarchy that can hold all tensors for generation. Avoid unnecessary offloading to slower storage.
- Tune the system to use a as large as possible batch size without out-of-memory.

| System | OPT-6.7B | OPT-30B | OPT-175B |
| ------ | -------- | ------- | -------- |
| Hugging Face Accelerate  | 2  (gpu) | 8 (cpu)   | 2 (disk)   |
| DeepSpeed ZeRO-Inference | 16 (cpu) | 4 (cpu)   | 1 (disk)   |
| FlexGen                  | 2  (gpu) | 144 (cpu) | 256 (disk) |
| FlexGen with Compression | 72 (gpu) | 512 (cpu) | 144 (cpu)  |

### Generation Throughput (token/s)
We attach the generation throughput here for reference.

| System | OPT-6.7B | OPT-30B | OPT-175B |
| ------ | -------- | ------- | -------- |
| Hugging Face Accelerate   | 25.12 | 0.62 | 0.01 |
| DeepSpeed ZeRO-Inference | 9.28  | 0.60 | 0.01 |
| FlexGen                  | 25.26 | 7.32 | 0.69 |
| FlexGen with Compression | **29.12** | **8.38** | **1.12** |

### About Petals
We also include [Petals](https://arxiv.org/abs/2209.01188) as an additional baseline.
We measure the results of running OPT hosted on 1, 4, and 24 T4 GPUs (in case of 6.7B, 30B, and 175B respectively) on GCP.
We perform 6 parallel requests to the system and divide the throughput by the number of used GPUs in each case.
For a more comprehensive comparison with Petals, see Section 6.3 in our paper.
