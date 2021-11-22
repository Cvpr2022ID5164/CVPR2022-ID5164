#!/bin/bash
n_gpu=2
cls='all'
tst_mdl=train_log/shapenet/checkpoints/all/FFB6D_all_best.pth.tar
python3 -m torch.distributed.launch --nproc_per_node=$n_gpu train_shapenet.py --gpus=$n_gpu --cls=$cls --deterministic -checkpoint $tst_mdl