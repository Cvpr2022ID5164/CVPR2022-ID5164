#!/bin/bash
tst_mdl=checkpoints/GAML_KNN8_all_PBR.pth.tar
for cls in airplane 
do 
    python3 -m torch.distributed.launch --nproc_per_node=1 train_shapenet.py --gpu '0' --cls $cls -eval_net -checkpoint $tst_mdl -test -test_pose  #-debug
done
