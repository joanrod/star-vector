#!/bin/bash

export HF_HOME=<path to the folder where you want to store the models>
export HF_TOKEN=<your huggingface token>
export WANDB_API_KEY=<your wandb token>
export OUTPUT_DIR=<path/to/output>

torchrun \
  --nproc-per-node=2 \
  --nnodes=1 \
  starvector/train/train.py \
  config=configs/models/starvector-8b/im2svg-stack.yaml
