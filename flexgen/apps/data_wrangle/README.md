# FlexGen for Data Wrangling Tasks.

Here we show how to use FlexGen for the data wrangling tasks. The implementation follows the [fm_data_tasks](https://github.com/HazyResearch/fm_data_tasks) repo from [HazyResearch](https://github.com/HazyResearch).

## Install

- Run our install.sh scripts to install additional python library and download the desired datasets.


## Examples

- To check the outcome and verify the result of a data imputation task (e.g., Restaurant on OPT-6.7B), run:

      bash test_single_query_case.sh

- To test FlexGen Throughput of a data imputation task (e.g., Restaurant on OPT-6.7B), run:

      bash test_batch_query_case.sh


## Benchmark Results

###  OPT6.7B 

| Task                   | Tested Samples    | Prompt Length | Output Length | Time (s) | Output Throughput | Total Throughput |
|------------------------|-------------------|---------------|---------------|----------|-------------------|------------------|
| EM: Fodors-Zagats      | 189               | 744           | 3             | 109.556  | 5.148             | 1281.871         |
| EM: Beer               | 91                | 592           | 3             | 42.087   | 6.415             | 1272.360         |
| EM: iTunes-Amazon      | 109               | 529           | 3             | 59.467   | 5.448             | 966.178          |
| EM: Walmart-Amazon     | 200 (from 2049)   | 748           | 3             | 126.538  | 4.742             | 1186.992         |
| EM: Amazon-Google      | 200 (from 2293)   | 876           | 3             | 144.593  | 4.150             | 1215.828         |
| EM: DBLP-ACM           | 200 (from 2473)   | 1274          | 3             | 207.513  | 2.891             | 1230.767         |
| EM: DBLP-GoogleScholar | 200 (from 5742)   | 1209          | 3             | 232.65   | 2.57              | 1097.78          |
| DI: Restaurant         | 86                | 123           | 5             | 10.397   | 38.471            | 984.865          |
| DI: Buy                | 65                | 488           | 10            | 43.077   | 14.857            | 739.876          |
| ED: Hospital           | 200 (from 17101)  | 200           | 3             | 30.137   | 19.909            | 1347.203         |


