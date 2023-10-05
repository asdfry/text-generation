#!/bin/bash

docker build -t asdfry/train-llm:gs .
yes | docker image prune
docker images
docker push asdfry/train-llm:gs
