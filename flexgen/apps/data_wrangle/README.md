# FlexGen for Data Wrangling Tasks.

Here we show how to use FlexGen for the data wrangling tasks including entity match (EM), data imputation (DI) and error detection (ED). The implementation follows the [fm_data_tasks](https://github.com/HazyResearch/fm_data_tasks) repo from [HazyResearch](https://github.com/HazyResearch).

## Install

- Run our install.sh scripts to install additional python library and download the desired datasets.


## Examples

- To check the outcome and verify the result of a data imputation task (e.g., Restaurant on OPT-6.7B), run:

      bash test_single_query_case.sh

- To test the throughput of FlexGen for a data imputation task (e.g., Restaurant on OPT-6.7B), run:

      bash test_batch_query_case.sh

- To run the complete tests of all tasks on OPT-6.7B:
  
      bash test_batch_query_all_opt6.7b.sh

- To run the complete tests of all tasks on OPT-30B:
  
      bash test_batch_query_all_opt30b.sh

- To run the complete tests of all tasks on OPT-175B:
  
      bash test_batch_query_all_opt175b.sh



## Benchmark Results

- Notice that in this data wrangling tasks, such as entity match (EM), data imputation (DI) and error detection (ED), the input sequences length is **very long** (from 123 to 1274), but the output length is **very short** (e.g., 3, 5, or 10). Most of the inference time is spent on prefill phase, so here we report the throughput that includes both input and output tokens as our measurement. 

- We run the experiments on the same setting as the HELM benchmark with a single T4 (16GB) GPU, 200GB of DRAM, and 1.5TB SSD connected by NVMe.

###  OPT6.7B 

| Task                   | Tested Samples    |  Input Length | Output Length | Time (s) |Input + Output Throughput (token/s)|
|------------------------|-------------------|---------------|---------------|----------|----------------------|
| EM: Fodors-Zagats      | 189               | 744           | 3             | 109.556  | 1281.871             |
| EM: Beer               | 91                | 592           | 3             | 42.087   | 1272.360             |
| EM: iTunes-Amazon      | 109               | 529           | 3             | 59.467   | 966.178              |
| EM: Walmart-Amazon     | 200               | 748           | 3             | 126.538  | 1186.992             |
| EM: Amazon-Google      | 200               | 876           | 3             | 144.593  | 1215.828             |
| EM: DBLP-ACM           | 200               | 1274          | 3             | 207.513  | 1230.767             |
| EM: DBLP-GoogleScholar | 200               | 1209          | 3             | 232.65   | 1097.78              |
| DI: Restaurant         | 86                | 123           | 5             | 10.397   | 984.865              |
| DI: Buy                | 65                | 488           | 10            | 43.077   | 739.876              |
| ED: Hospital           | 200               | 200           | 3             | 30.137   | 1347.203             |


###  OPT30B 

| Task                   | Tested Samples    |  Input Length | Output Length | Time (s) |Input + Output Throughput (token/s)|
|------------------------|-------------------|---------------|---------------|----------|----------------------|
| EM: Fodors-Zagats      | 189               | 744           | 3             | 541.550  | 248.287              |
| EM: Beer               | 91                | 592           | 3             | 238.58   | 224.450              |
| EM: iTunes-Amazon      | 109               | 529           | 3             | 267.639  | 198.775              |
| EM: Walmart-Amazon     | 200               | 748           | 3             | 682.635  | 220.030              |
| EM: Amazon-Google      | 200               | 876           | 3             | 799.514  | 219.884              |
| EM: DBLP-ACM           | 200               | 1274          | 3             | 1119.272 | 228.184              |
| EM: DBLP-GoogleScholar | 200               | 1209          | 3             | 1271.534 | 190.636              |
| DI: Restaurant         | 86                | 123           | 5             | 60.310   | 169.790              |
| DI: Buy                | 65                | 488           | 10            | 185.882  | 160.747              |
| ED: Hospital           | 200               | 200           | 3             | 158.329  | 256.429              |


###  OPT175B 

| Task                   | Tested Samples |  Input Length | Output Length | Time (s) |Input + Output Throughput (token/s)|
|------------------------|----------------|---------------|---------------|----------|----------------------|
| EM: Fodors-Zagats      | 189            | 744           | 3             |3928.310  | 34.228               |
| EM: Beer               | 91             | 592           | 3             |1356.786  | 35.083               |
| EM: iTunes-Amazon      | 109            | 529           | 3             |1569.062  | 33.906               |
| EM: Walmart-Amazon     | 200            | 748           | 3             |4171.319  | 36.008               |
| EM: Amazon-Google      | 200            | 876           | 3             |4893.572  | 35.925               |
| EM: DBLP-ACM           | 200            | 1274          | 3             |7624.726  | 33.496               |
| EM: DBLP-GoogleScholar | 200            | 1209          | 3             |8275.828  | 29.290               |
| DI: Restaurant         | 86             | 123           | 5             |648.762   | 16.968               |
| DI: Buy                | 65             | 488           | 10            |2086.961  | 14.317               |
| ED: Hospital           | 200            | 200           | 3             |1154.133  | 35.178               |
