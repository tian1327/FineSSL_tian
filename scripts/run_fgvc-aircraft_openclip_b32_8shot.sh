#!/bin/bash

echo "Starting Semi-Aves training run..."
echo "Running on host: $(hostname)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "Current directory: $(pwd)"
echo "Using config: configs/fgvc-aircraft_openclip_b32_8shot.yaml"
echo "Time: $(date)"


# Run training script
python -u main.py --cfg configs/peft/fgvc-aircraft_openclip_b32_8shot.yaml


echo "Training completed at: $(date)"
