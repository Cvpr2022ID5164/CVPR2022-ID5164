#!/bin/bash
n_gpu=1
cls='airplane'
python3 -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$RANDOM train_shapenet.py --gpus=$n_gpu --cls=$cls --deterministic
