# Running Petals benchmarks

This guide contains the steps necessary to reproduce experiments in Section 6.3 and Table 17 of the paper.

## Requirements

To start the benchmarks, you will need a Linux-based environment with an installed [version of Petals that supports OPT-sized models](https://github.com/bigscience-workshop/petals/tree/test_opt_serving).
We provide a Docker image with such an environment at [mrbn/petals:test_opt_serving](https://hub.docker.com/layers/mrbn/petals/test_opt_serving/images/sha256-5c38f459f1b42fc655f85523b78c22a1b3c05139d9bd03cd5e2a395e8d73b7aa?context=explore) on DockerHub.
You will also need to install the [Traffic Control](https://wiki.debian.org/TrafficControl) (`tc`) utility for controlling the network connection speed.

## Setting up a private swarm

First, you need to set up a coordinator peer using [this part](https://github.com/bigscience-workshop/petals/wiki/Launch-your-own-swarm#step-1-set-up-the-network) of the Petals private swarm guide.
If you do not need a persistent identifier, a single command might suffice:

```
hivemind-dht
```
After the DHT node is started, it will give you an address of this peer. 
You will need it for connecting all other servers and clients to each other: following parts of the guide assume it is stored in the `$INITIAL_PEER_ID` environment variable.

Then, on each of the machines with GPUs, you need to run two commands that set the network throughput and latency and launch the actual server. 
We assume OPT-30B model in these commands (Section 6.3); replace the model names with `facebook/opt-6.7b` or `facebook/opt-175b` if necessary.

The commands for each setup are below. 
If you are not using Docker, simply remove everything before `python -m petals.cli.run_server`.
Also, change `$NETWORK_INTERFACE` in the first command to the external [network interface](https://www.cyberciti.biz/faq/linux-list-network-interfaces-names-command/) used by your GPU nodes.

* 10ms latency, 1 Gbit throughput
    ```
    tc qdisc add dev $NETWORK_INTERFACE root netem delay 10ms rate 1Gbit limit 225000
    sudo docker run --net host --ipc host --gpus all --name petals --volume petals-cache:/cache \
      --rm mrbn/petals:test_opt_serving python -m petals.cli.run_server facebook/opt-30b --initial_peers $INITIAL_PEER_ID
    ```
* 10ms latency, 100 Mbit throughput
    ```
    tc qdisc add dev $NETWORK_INTERFACE root netem delay 10ms rate 0.1Gbit limit 22500
    sudo docker run --net host --ipc host --gpus all --name petals --volume petals-cache:/cache \
      --rm mrbn/petals:test_opt_serving python -m petals.cli.run_server facebook/opt-30b --initial_peers $INITIAL_PEER_ID
    ```
* 100ms latency, 100Mbit throughput
    ```
    tc qdisc add dev $NETWORK_INTERFACE root netem delay 100ms rate 0.1Gbit limit 2250000
    sudo docker run --net host --ipc host --gpus all --name petals --volume petals-cache:/cache \
      --rm mrbn/petals:test_opt_serving python -m petals.cli.run_server facebook/opt-30b --initial_peers $INITIAL_PEER_ID
    ```

## Running benchmarks

Finally, on another GPU-enabled machine (preferably with at least 2 GPUs) with the same environment, run the following command:

```
python run_opt_requests.py --initial_peers $INITIAL_PEER --prefix facebook/opt-30b \
  -b 1 --num-micro-batches 2 --num-processes 6 --output out_30b.tsv
```

This script will produce a TSV file with the following columns:
* Microbatch size
* Number of microbatches
* Number of processes
* Prefix length
* Output sequence length
* Total throughput (**needs to be divided by the number of GPU peers in the swarm**)
* Average latency per token

Using this data, you can reproduce the results from our paper by either displaying results for several batch sizes in a table (Table 17) or plotting the throughput/latency trends with respect to the number of generated tokens.