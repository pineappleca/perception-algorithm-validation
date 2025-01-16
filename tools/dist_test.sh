#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
# GPUS=$3
# PORT=${PORT:-29503}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/test.py $CONFIG $CHECKPOINT ${@:4} --eval bbox --corruption '{"add_rain": 5.938429355621338, "sensor_gnoise": 5.017270088195801, "camera_blur": 5.994455337524414}'

