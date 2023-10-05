#!/bin/bash

accelerate launch --config_file config.yaml \
train.py -b 2 -d 1.0 -e 1 -mn bigscience/bloom-560m
