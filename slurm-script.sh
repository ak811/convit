#!/bin/bash
#SBATCH --job-name="MAE-FineTune"
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=32GB
#SBATCH --time=24:30:00

module load anaconda3
module load cuda

conda activate convitenv

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 26908 --use_env main.py --model convit_tiny --batch-size 16 \
      --output_dir exp/cvt-tiny/c10   --data-path ./data --data-set CIFAR10  \



