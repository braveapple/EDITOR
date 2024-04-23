#!/bin/bash

set -e

CUDA_VISIBLE_DEVICES='1' \
    python train_net.py \
    --config_file configs/RGBNT201/EDITOR.yml \
    SOLVER.IMS_PER_BATCH 64
