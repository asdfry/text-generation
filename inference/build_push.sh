#!/bin/bash

if [ $# = 0 ]; then
    echo "Date argument required"
elif [ $# = 1 ]; then
    docker build -t asdfry/infer-llm:$1 .
    yes | docker image prune
    docker images
    docker push asdfry/infer-llm:$1
fi
