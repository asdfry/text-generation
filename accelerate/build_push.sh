#!/bin/bash

if [ $# = 0 ]; then
    echo "Date argument required"
elif [ $# = 1 ]; then
    docker build -t asdfry/train-llm:$1 .
    yes | docker image prune
    docker images
    docker push asdfry/train-llm:$1
fi
