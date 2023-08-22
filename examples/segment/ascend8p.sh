#!/bin/bash

export DEVICE_NUM=8
export RANK_SIZE=8
export RANK_TABLE_FILE="/home/cvgroup/wcr/mindcv_latest/hccl.json"

for ((i = 0; i < ${DEVICE_NUM}; i++)); do
   export DEVICE_ID=$i
   export RANK_ID=$i
   python -u train.py --config test.yaml  &> ./train_$i.log &
done
