#!/bin/bash

docker build -t asdfry/train-llm:add .
yes | docker image prune
docker images
docker push asdfry/train-llm:add
