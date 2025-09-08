#!/bin/bash

# DeepSpeed training script for optimal performance on 8 B200s
export WANDB_PROJECT="gemma3-cpt"
export TOKENIZERS_PARALLELISM=false

# Number of GPUs
NUM_GPUS=8

# Run with DeepSpeed
deepspeed --num_gpus=$NUM_GPUS \
    --master_port=29500 \
    train.py \
    --model_id google/gemma-3-12b-it \
    --vision_model_id google/siglip-so400m-patch14-384 \
    --dataset_name OpenGVLab/OmniCorpus-CC-210M \
    --output_dir ./outputs \
    --batch_size 4096 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --warmup_steps 100 \
    --logging_steps 10 \
    --save_steps 500 \
    --seed 42 \
    --use_deepspeed