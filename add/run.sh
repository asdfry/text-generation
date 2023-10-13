#!/bin/bash

accelerate launch --config_file config.yaml \
train.py -a -b 8 -d 0.5 -e 1 -mn bloom-1b1
