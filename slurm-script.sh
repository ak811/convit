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






#conda create --name convitenv  python=3.7 -y
conda activate convitenv


#pip install timm==0.3.2 torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
#pip install matplotlib timm einops tensorboard
#--datapath /users/mgovind/data/imagenet-tiny \
# Install required packages


# CUDA_VISIBLE_DEVICES=0,1, python -m torch.distributed.launch --nproc_per_node 2 --master_port 26523 main_finetune.py \
#     --dataset c10 --model vit_base_patch16 \
#     --data_path /users/mgovind/data/imagenet-tiny \
#     --epochs 100 \
#     --cls_token \
#     --nb_classes 10 \
#     --batch_size 32 \
#     --output_dir exp/c10/multiheadwindow/246 \
#     --log_dir  exp/c10/multiheadwindow/246 \
#     --blr 1e-3 --layer_decay 0.75 \
#     --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mask_ratio 0.9 --num_workers 8



CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 26908 --use_env main.py --model convit_tiny --batch-size 16 \
      --output_dir exp/cvt-tiny/c10   --data-path ./data --data-set CIFAR10  \



