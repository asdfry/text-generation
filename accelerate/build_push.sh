#!/bin/bash
docker build -t asdfry/train-llm:$1 .
yes | docker image prune
docker images
docker push asdfry/train-llm:$1
