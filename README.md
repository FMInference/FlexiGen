# FlexGen

FlexGen is a high-throughput generation engine for running large language models with limited GPU memory. FlexGen allows **high-throughput** generation by IO-efficient offloading, compression, and **large effective batch sizes**.

## Throughput-Oriented Inference for Large Language Models

In recent years, large language models (LLMs) have shown great performance across a 
wide range of tasks. Increasingly, LLMs have been applied not only to interactive 
applications (such as chat), but also to many "back-of-house" tasks.
These tasks include benchmarking, information extraction, data wrangling, and form processing.

One key characteristic of these applications is that they are **throughput-oriented**: they require
running LLM inferences over millions of tokens in batches, e.g., all the private documents in a company's
corpus, or all the tasks in the [HELM](https://crfm.stanford.edu/helm/latest/) benchmark.
These workloads are less sensitive to latency - the user starts up a job and lets it run overnight -
but increasing throughput is critical for reducing costs.
Throughput is a measure of tokens processed per second over the job's entire runtime (which can be hours).
Throughput-oriented workloads provide opportunities to trade off latency for higher throughput, which
makes it easier to take advantage of low-cost commodity GPUs. 

The goal of FlexGen is to create a high-throughput system to enable new and exciting applications of 
foundation models to throughput-oriented tasks on low-cost hardware, such as a single commodity GPU
instead of expensive systems.

Check out the [examples](#examples) of what you can run on a single commodity GPU with FlexGen, including benchmarking and data wrangling.

❌ **Limitation**. As an offloading-based system running on weak GPUs, FlexGen also has its limitations.
FlexGen can be significantly slower than the case when you have enough powerful GPUs to hold the whole model, especially for small-batch cases.
FlexGen is mostly optimized for throughput-oriented batch processing settings (e.g., classifying or extracting information from many documents in batches), on single GPUs.

----------

This project was made possible thanks to a collaboration with

<a href="https://cs.stanford.edu/"><img src="https://identity.stanford.edu/wp-content/uploads/sites/3/2020/06/wordmark-nospace-red.png" height="20"></a> &nbsp;&nbsp;&nbsp;
<a href="https://sky.cs.berkeley.edu/"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/82/University_of_California%2C_Berkeley_logo.svg/1280px-University_of_California%2C_Berkeley_logo.svg.png" height="22"></a> &nbsp;&nbsp;&nbsp;
<a href="https://www.andrew.cmu.edu/user/beidic/"><img src="https://upload.wikimedia.org/wikipedia/commons/9/9b/Carnegie_Mellon_wordmark.svg" height="20"></a> &nbsp;&nbsp;&nbsp;
<a href="https://www.together.xyz/"><img src="https://images.squarespace-cdn.com/content/v1/6358bea282189a0adf57fe16/eef09191-631f-40d9-9bfd-f875b25bcf0b/together-logo-black-transparent2.png" height="20"></a> &nbsp;&nbsp;&nbsp;
<a href="https://research.yandex.com/"><img src="https://storage.yandexcloud.net/yandex-research/assets/yandex_research.png" height="20"></a> &nbsp;&nbsp;&nbsp;
<a href="https://ds3lab.inf.ethz.ch/"><img src="https://user-images.githubusercontent.com/1608867/220273382-c09669b3-42fd-47c2-b88c-7ed55cb43820.png" height="20"></a>

----------

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

## Examples 
### HELM Benchmark 
FlexGen can be integrated into [HELM](https://crfm.stanford.edu/helm), a language model benchmark framework, as its execution backend.
You can use the commands below to run a Massive Multitask Language Understanding (MMLU) [scenario](https://crfm.stanford.edu/helm/latest/?group=mmlu) with a single T4 (16GB) GPU and 200GB of DRAM.
```
python3 -m flexgen.apps.helm_run --description mmlu:model=text,subject=abstract_algebra,data_augmentation=canonical --pad-to-seq-len 512 --model facebook/opt-30b --percent 20 80 0 100 0 100 --gpu-batch-size 48 --num-gpu-batches 3 --max-eval-instance 100
```
Note that only a subset of HELM scenarios is tested. See more tested scenarios [here](flexgen/apps/helm_passed_30b.sh).

### Data Wrangling
You can run the examples in this paper, ['Can Foundation Models Wrangle Your Data?'](https://arxiv.org/abs/2205.09911), by following the instructions [here](flexgen/apps/data_wrangle).

## Performance Benchmark
### Generation Throughput (token/s)
The corresponding effective batch sizes and lowest offloading devices are in parentheses. Please see [here](benchmark/batch_size_table.md) for more details.
| System | OPT-6.7B | OPT-30B | OPT-175B |
| ------ | -------- | ------- | -------- |
| Hugging Face Accelerate  | 25.12 (2 on GPU)  | 0.62 (8 on CPU) | 0.01 (2 on disk) |
| DeepSpeed ZeRO-Inference | 9.28 (16 on CPU)  | 0.60 (4 on CPU) | 0.01 (1 on disk) |
| Petals                 | 8.25 (2 on GPU) | 2.84 (2 on GPU) | 0.08 (2 on GPU) |
| FlexGen                  | 25.26 (2 on GPU) | 7.32 (144 on CPU) | 0.69 (256 on disk) |
| FlexGen with Compression | **29.12** (72 on GPU) | **8.38** (512 on CPU) | **1.12** (144 on CPU) |

- Hardware: an NVIDIA T4 (16GB) instance on GCP with 208GB of DRAM and 1.5TB of SSD.  
- Workload: input sequence length = 512, output sequence length = 32. The batch size is tuned to **a large value** that maximizes the generation throughput for each system.
- Metric: generation throughput (token/s) = number of the generated tokens / (time for processing prompts + time for generation).  

How to [reproduce](benchmark/flexgen).

## Roadmap
We plan to work on the following features.

- [ ] Optimize the performance for multiple GPUs on the same machine
- [ ] Support more models (BLOOM, CodeGen, GLM)
- [ ] Release the cost model and policy optimizer
- [ ] Macbook Support (M1 and M2)
- [ ] AMD Support
