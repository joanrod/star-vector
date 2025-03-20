#!/bin/bash

python starvector/validation/run_validator.py \
config=configs/generation/starvector-8b/im2svg.yaml \
dataset.name svg-stack \ 
model.generation_engine=hf

python starvector/validation/run_validator.py \
config=configs/generation/starvector-8b/im2svg.yaml \
dataset.name svg-emoji \ 
model.generation_engine=hf

python starvector/validation/run_validator.py \
config=configs/generation/starvector-8b/im2svg.yaml \
dataset.name svg-fonts \ 
model.generation_engine=hf

python starvector/validation/run_validator.py \
config=configs/generation/starvector-8b/im2svg.yaml \
dataset.name svg-diagrams \ 
model.generation_engine=hf

python starvector/validation/run_validator.py \
config=configs/generation/starvector-8b/im2svg.yaml \
dataset.name svg-icons \ 
model.generation_engine=hf


