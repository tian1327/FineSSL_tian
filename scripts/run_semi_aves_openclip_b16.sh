#!/bin/bash

echo "Starting Semi-Aves training run..."
echo "Running on host: $(hostname)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "Current directory: $(pwd)"
echo "Using config: configs/semi_aves_openclip_b16.yaml"
echo "Time: $(date)"


# Run training script
python -u main.py --cfg configs/peft/semi_aves_openclip_b16.yaml


echo "Training completed at: $(date)"
