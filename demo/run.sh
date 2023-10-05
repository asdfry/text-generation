#!/bin/bash

NCCL_DEBUG=INFO \
NCCL_TOPO_DUMP_FILE=nccl-topo.xml \
accelerate launch --config_file config.yaml \
train.py -b 16 -e 1 -mn bigscience/bloom-560m
