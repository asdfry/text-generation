#!/bin/bash

accelerate launch --config_file config.yaml train.py -b 2 -e 1 -t
