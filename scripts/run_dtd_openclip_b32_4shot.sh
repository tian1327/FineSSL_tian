#!/bin/bash

echo "Starting Semi-Aves training run..."
echo "Running on host: $(hostname)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "Current directory: $(pwd)"
echo "Using config: configs/dtd_openclip_b32_4shot.yaml"
echo "Time: $(date)"


# Run training script
python -u main.py --cfg configs/peft/dtd_openclip_b32_4shot.yaml


echo "Training completed at: $(date)"
