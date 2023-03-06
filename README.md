# Motivation: Throughput-Oriented Systems
This project focuses on throughput-oriented large language model (LLM) generative inference when limited GPU memory is available.
LLMs are getting used in new tasks where many inputs can be batched together, such as benchmarking, information extraction, data wrangling, and form processing, but foundational models have not had a massive impact on them mainly due to limitations on computational resources and privacy concerns.
For example, in the real world, data from such tasks can be sensitive and confidential, such as data from hospitals, banks, and funds. As computation infrastructures in such organizations are often outdated, expensive GPU resources are scarce.
The above motivates us to initiate this study of high-throughput LLM inference with limited resources.
Another reason for targeting high throughput is that we can significantly increase throughput by trading off latency, especially in memory-limited cases, and the tasks we mentioned above are usually not latency sensitive.

The goal of this project is to create a high-throughput system to enable new and exciting applications of foundational models in these tasks on low-cost hardware, such as a single commodity GPU, instead of expensive systems.
We demonstrate an example use case on the [HELM](https://crfm.stanford.edu/helm) benchmark.

----------

This project was made possible thanks to a collaboration with

<a href="https://cs.stanford.edu/"><img src="https://identity.stanford.edu/wp-content/uploads/sites/3/2020/06/wordmark-nospace-red.png" height="20"></a> &nbsp;&nbsp;&nbsp; <a href="https://sky.cs.berkeley.edu/"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/82/University_of_California%2C_Berkeley_logo.svg/1280px-University_of_California%2C_Berkeley_logo.svg.png" height="22"></a> &nbsp;&nbsp;&nbsp; <a href="https://www.andrew.cmu.edu/user/beidic/"><img src="https://upload.wikimedia.org/wikipedia/commons/9/9b/Carnegie_Mellon_wordmark.svg" height="20"></a> &nbsp;&nbsp;&nbsp; <a href="https://www.together.xyz/"><img src="https://images.squarespace-cdn.com/content/v1/6358bea282189a0adf57fe16/eef09191-631f-40d9-9bfd-f875b25bcf0b/together-logo-black-transparent2.png" height="20"></a> &nbsp;&nbsp;&nbsp; <a href="https://ds3lab.inf.ethz.ch/"><img src="https://user-images.githubusercontent.com/1608867/220273382-c09669b3-42fd-47c2-b88c-7ed55cb43820.png" height="20"></a>

----------

# FlexGen (Still Working in Progress!)
FlexGen is a high-throughput generation engine for running large language models with limited GPU memory. FlexGen allows **high-throughput** generation by IO-efficient offloading, compression and **large effective batch sizes**.

⚡ **High-Throughput Offloading**.  
Higher-throughput generation than other offloading-based systems (e.g., Hugging Face Accelerate, DeepSpeed Zero-Inference) - sometimes by orders of magnitude. This can be useful for batch inference scenarios, such as benchmarking (e.g., [HELM](https://github.com/stanford-crfm/helm)) and [data wrangling](https://arxiv.org/abs/2205.09911).

❌ **Limitation**.  
As an offloading-based system running on weak GPUs, FlexGen also has its limitations.
FlexGen can be significantly slower than the case when you have enough powerful GPUs to hold the whole model, especially for small-batch cases.
FlexGen is mostly optimized for throughput-oriented batch processing settings (e.g., classifying or extracting information from many documents in batches), on single GPUs.

## Install
Requirements:  
 - PyTorch >= 1.12 [(Help)](https://pytorch.org/get-started/locally/)

### Method 1: With pip
```
pip install flexgen
```

### Method 2: From source
```
git clone https://github.com/FMInference/FlexGen.git
cd FlexGen
pip install -e .
```

## HELM Benchmark Example
FlexGen can be integrated into [HELM](https://crfm.stanford.edu/helm), a language model benchmark framework, as its execution backend.
You can use the commands below to run a Massive Multitask Language Understanding (MMLU) [scenario](https://crfm.stanford.edu/helm/latest/?group=mmlu) with a single T4 (16GB) GPU and 200GB of DRAM.
```
python3 -m flexgen.apps.helm_run --description mmlu:model=text,subject=abstract_algebra,data_augmentation=canonical --pad-to-seq-len 512 --model facebook/opt-30b --percent 20 80 0 100 0 100 --gpu-batch-size 48 --num-gpu-batches 3 --max-eval-instance 100
```

## Performance Benchmark
### Generation Throughput (token/s)
The corresponding effective batch sizes are in the bracket. Please see [here](benchmark/batch_size_table.md) for more details.
| System | OPT-6.7B | OPT-30B | OPT-175B |
| ------ | -------- | ------- | -------- |
| Hugging Face Accelerate   | 25.12 (2 on gpu) | 0.62 (8 on cpu	) | 0.01 (2 on disk) |
| DeepSpeed ZeRO-Inference | 9.28 (16 on cpu)  | 0.60 (4 on cpu) | 0.01 (1 on disk) |
| Petals                 | 8.25     | 2.84    | 0.08 |
| FlexGen                  | 25.26 (2 on gpu) | 7.32 (144 on cpu) | 0.69 (256 on disk) |
| FlexGen with Compression | **29.12** (72 on gpu) | **8.38** (512 on cpu) | **1.12** (144 on cpu) |

- Hardware: an NVIDIA T4 (16GB) instance on GCP with 208GB of DRAM and 1.5TB of SSD.  
- Workload: input sequence length = 512, output sequence length = 32. The batch size is tuned to **a large value** that maximizes the generation throughput for each system.
- Metric: generation throughput (token/s) = number of the generated tokens / (time for processing prompts + time for generation).  

How to [reproduce](benchmark/flexgen).

## Roadmap
We plan to work on the following features.

- [ ] Optimize the performance for multip-GPUs on the same machine 
- [ ] Support more models (BLOOM, CodeGen, GLM)
- [ ] Release the cost model and policy optimizer
