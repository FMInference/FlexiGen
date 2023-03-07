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
| EM: Walmart-Amazon     | 200               | 748           | 3             | 126.538  | 4.742             | 1186.992         |
| EM: Amazon-Google      | 200               | 876           | 3             | 144.593  | 4.150             | 1215.828         |
| EM: DBLP-ACM           | 200               | 1274          | 3             | 207.513  | 2.891             | 1230.767         |
| EM: DBLP-GoogleScholar | 200               | 1209          | 3             | 232.65   | 2.57              | 1097.78          |
| DI: Restaurant         | 86                | 123           | 5             | 10.397   | 38.471            | 984.865          |
| DI: Buy                | 65                | 488           | 10            | 43.077   | 14.857            | 739.876          |
| ED: Hospital           | 200               | 200           | 3             | 30.137   | 19.909            | 1347.203         |


###  OPT30B 

| Task                   | Tested Samples    | Prompt Length | Output Length | Time (s) | Output Throughput | Total Throughput |
|------------------------|-------------------|---------------|---------------|----------|-------------------|------------------|
| EM: Fodors-Zagats      | 189               | 744           | 3             | 541.550  | 0.997             | 248.287          |
| EM: Beer               | 91                | 592           | 3             | 238.58   | 1.130             | 224.450          |
| EM: iTunes-Amazon      | 109               | 529           | 3             | 267.639  | 1.121             | 198.775          |
| EM: Walmart-Amazon     | 200               | 748           | 3             | 682.635  | 0.879             | 220.030          |
| EM: Amazon-Google      | 200               | 876           | 3             | 799.514  | 0.750             | 219.884          |
| EM: DBLP-ACM           | 200               | 1274          | 3             | 1119.272 | 0.536             | 228.184          |
| EM: DBLP-GoogleScholar | 200               | 1209          | 3             | 1271.534 | 0.472             | 190.636          |
| DI: Restaurant         | 86                | 123           | 5             | 60.310   | 6.632             | 169.790          |
| DI: Buy                | 65                | 488           | 10            | 185.882  | 3.228             | 160.747          |
| ED: Hospital           | 200               | 200           | 3             | 158.329  | 3.790             | 256.429          |


###  OPT175B 

| Task                   | Tested Samples | Prompt Length | Output Length | Time (s) | Output Throughput | Total Throughput |
|------------------------|----------------|---------------|---------------|----------|-------------------|------------------|
| EM: Fodors-Zagats      | 189            | 744           | 3             |3928.310  | 0.137             | 34.228           |
| EM: Beer               | 91             | 592           | 3             |1356.786  | 0.177             | 35.083           |
| EM: iTunes-Amazon      | 109            | 529           | 3             |          |                   |                  |
| EM: Walmart-Amazon     | 200            | 748           | 3             |          |                   |                  |
| EM: Amazon-Google      | 200            | 876           | 3             |          |                   |                  |
| EM: DBLP-ACM           | 200            | 1274          | 3             |          |                   |                  |
| EM: DBLP-GoogleScholar | 200            | 1209          | 3             |          |                   |                  |
| DI: Restaurant         | 86             | 123           | 5             |          |                   |                  |
| DI: Buy                | 65             | 488           | 10            |          |                   |                  |
| ED: Hospital           | 200            | 200           | 3             |          |                   |                  |