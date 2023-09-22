#!/bin/bash
docker build -t asdfry/train-llm:$1
docker push asdfry/train-llm:$1
