#!/bin/bash

N_GPUS=1
N_NODES=4
N_CORES_PER_GPU=16

MY_IPADDR=$(hostname -i)
all_public_ips=$(ray get-worker-ips ~/ray_bootstrap_config.yaml)
for s in $all_public_ips; do
    ssh -o StrictHostKeyChecking=no $s hostname -i > /tmp/$s.ip &
done
wait
for s in $all_public_ips; do
    OTHERS_IPADDR+=($(cat /tmp/$s.ip))
done
ALL_IPADDR=($MY_IPADDR ${OTHERS_IPADDR[@]})
all_hosts=$(echo ${ALL_IPADDR[@]:0:$N_NODES} | sed 's/ /,/g')

PYTHON_EXEC=$CONDA_PREFIX/bin/python
PYTHON_SCRIPT=flexgen.dist_flex_opt

pgrep -fl python | awk '!/dist_flex_opt\.py/{print $1}' | xargs sudo kill

set -x

mpirun \
  --mca btl_tcp_if_exclude lo,docker0 \
  --mca oob_tcp_if_exclude lo,docker0 \
  --map-by ppr:$N_GPUS:node:pe=$N_CORES_PER_GPU --oversubscribe -H $all_hosts \
  --bind-to core -x OMP_NUM_THREADS=$N_CORES_PER_GPU \
  $PYTHON_EXEC -m $PYTHON_SCRIPT \
    --head-ip $MY_IPADDR \
    --port 7777 \
    --use-mpi \
    --model facebook/opt-175b \
    --gpu-batch-size 40 \
    --num-inner-iterations 4 \
    --percent 0 100 0 100 0 100 \
    --comm-device cpu \
    --path _DUMMY_ \
    --cut-gen-len 5 \
    --pin-weight 0 \
    --cpu \
    --async-comm
