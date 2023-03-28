#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-17600}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    $(dirname "$0")/da_train.py $CONFIG --resume-from /GOODDATA/lihaochen/DSS/fin.pth --launcher pytorch ${@:3}

#python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    $(dirname "$0")/da_train.py $CONFIG --resume-from /GOODDATA/lihaochen/DSS/city_epoch_17.pth --launcher pytorch ${@:3}

#python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    $(dirname "$0")/da_train.py $CONFIG --launcher pytorch ${@:3}

#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    $(dirname "$0")/da_train.py $CONFIG --resume-from /GOODDATA/lihaochen/DSS/city_to_foggy_multi_dis_res50_49.6_mAP.pth --launcher pytorch ${@:3}
python  $(dirname "$0")/da_train.py $CONFIG #--resume-from /GOODDATA/lihaochen/detpro/work_dirs/da_cityscapes_cityscapes/epoch_79.pth