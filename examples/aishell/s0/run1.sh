#!/bin/bash
set -x

echo $*

args=`echo $SM_USER_ARGS | jq -r 'join(" ")'`

cd /fsx/wenet/examples/aishell/s0

bash run.sh $args

echo "RUN whereis smddprun"
whereis smddprun

# ls /opt/ml/input/data/wav/train/S0002/BAC009S0002W0123.wav
# head /opt/ml/input/data/train/wav.scp

export CUDA_VISIBLE_DEVICES="0"

trail_dir=
train_set=

# cd /opt/ml/code/examples/aishell/s0

. ./path.sh || exit 1;
# . tools/parse_options.sh || exit 1;

# echo $CUDA_VISIBLE_DEVICES

# echo $trail_dir


# ls -R /opt/ml/input/data

# ls -R /fsx

# sleep 100000