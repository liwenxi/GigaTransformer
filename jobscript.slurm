#!/bin/bash
#SBATCH --job-name=Giga768
#SBATCH -p a100
#SBATCH --output=%j_OHEM_train.out
#SBATCH -e %j_OHEM_train.err
#SBATCH -n 1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2

source activate swin
ulimit -SHn 51200
export PYTHONPATH=$(pwd)
python -m torch.distributed.launch --nproc_per_node 2 --master_port 12121  main.py --cfg configs/giga_patch16_stride6_window8_768.yaml --data-path /dssg/home/acct-eexdl/eexdl/liwenxi/Camelyon16-Zip/ --batch-size 256
