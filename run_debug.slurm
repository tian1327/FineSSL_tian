#!/bin/bash

## ENVIRONMENT SETTINGS
#SBATCH --get-user-env=L          # Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=semi_aves_debugging
#SBATCH --output=slurm_log/%x.%j.out
#SBATCH --error=slurm_log/%x.%j.err
##SBATCH --account=132713661936
#SBATCH --time=1:00:00
#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=12
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH --gres=gpu:a100:1             # Request 1 GPUs per node
##SBATCH --gres=gpu:rtx:1 
##SBATCH --gres=gpu:t4:1
#SBATCH --partition=gpu          # Submit job to the gpu queue

##SBATCH --mail-type=ALL 
##SBATCH --mail-user=anwesha.basu@tamu.edu

python -u dry_run_debug.py

##----------- Job Environment variables
echo "Job ID: $SLURM_JOBID"
echo "Job submission directory: $SLURM_SUBMIT_DIR"
echo "Temp directory: $TMPDIR"
echo "Scratch directory: $SCRATCH"


