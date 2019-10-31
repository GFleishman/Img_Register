#!/bin/bash

GREEDY_PATH=/groups/scicompsoft/home/dingx/Documents/GANs/greedy/build
PHI_PATH=/nrs/scicompsoft/dingx/GAN_data/toy_data
OUT_PATH=/nrs/scicompsoft/dingx/GAN_data/toy2

for fold in `ls -F $PHI_PATH | grep \/$`; do
    echo "processing $fold"
    mkdir -p $OUT_PATH/$fold
    $GREEDY_PATH/greedy -threads 16 -d 3 -rf $OUT_PATH/sphere.nrrd -rm $OUT_PATH/sphere.nrrd $OUT_PATH/$fold/warped.nrrd -r $PHI_PATH/$fold/phiinv.nrrd
done 