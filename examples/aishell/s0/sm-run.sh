#!/bin/bash

cd /opt/ml/code/examples/aishell/s0

args=`echo $SM_USER_ARGS | jq -r 'join(" ")'`
bash run.sh $args