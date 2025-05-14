#!/bin/bash

echo "Starting Semi-Aves training run..."
echo "Running on host: $(hostname)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "Current directory: $(pwd)"
echo "Using config: configs/eurosat_openclip_b32_16shot.yaml"
echo "Time: $(date)"


# Run training script
python -u main.py --cfg configs/peft/eurosat_openclip_b32_16shot.yaml


echo "Training completed at: $(date)"
