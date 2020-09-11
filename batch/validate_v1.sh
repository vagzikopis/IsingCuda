#!/bin/bash
#SBATCH --job-name=v1
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=1:00


../src/./validate_v1
