#!/usr/bin/env bash
# 使用./tools/test.py生成的.pkl文件进行算法eval
# 避免内存溢出事件

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29503}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test_from_json.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox