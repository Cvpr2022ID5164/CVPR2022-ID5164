#!/bin/bash

# Training classes
# airplane bag bathtub bed bench bookshelf bus cabinet camera cap chair earphone motorcycle mug table train vessel washer printer

# Test classes
# birdhouse car piano laptop sofa

tst_mdl=checkpoints/GAML_KNN8_all_PBR.pth.tar
for cls in airplane chair sofa
do 
    python3 -m torch.distributed.launch --nproc_per_node=1 demo_parallel.py -dataset shapenet -checkpoint $tst_mdl -cls $cls
done