#!/bin/bash

if [ -z "$SM_CURRENT_HOST" ]; then
    export SM_CURRENT_HOST=`hostname`
    cd /fsx/wenet/examples/aishell/s0
else
    cd /opt/ml/code/examples/aishell/s0
fi

bash run.sh $*