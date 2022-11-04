#!/bin/bash

if [ -z "$SM_CURRENT_HOST" ]; then
    export SM_CURRENT_HOST=`hostname`
    cd /fsx/wenet/examples/gigaspeech/s0
else
    cd /opt/ml/code/examples/gigaspeech/s0
fi

bash run.sh $*