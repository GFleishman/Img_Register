#!/bin/bash

GREEDY_PATH=/groups/scicompsoft/home/dingx/Documents/GANs/greedy/build


# # Greedy on toy data
# TEMPLATE=/nrs/scicompsoft/dingx/GAN_data/data_fly/JRC2018_lo_masked.nrrd
# INPUT_DIR=/nrs/scicompsoft/dingx/GAN_data/data_fly/align_greedy
#
# $GREEDY_PATH/greedy -threads 16 -d 3 \
#     -s 2mm 1mm \
#     -m NCC 2x2x2 \
#     -i $TEMPLATE $INPUT_DIR/20170301_31_B5_Scope_1_C1_down_result_masked.nrrd \
#     -o $INPUT_DIR/20170301_31_B5_warp.nrrd \
#     -n 100x50x10
#
# $GREEDY_PATH/greedy -threads 16 -d 3 \
#     -rf $TEMPLATE \
#     -r $INPUT_DIR/20170301_31_B5_warp.nrrd \
#     -rm $INPUT_DIR/20170301_31_B5_Scope_1_C1_down_result_masked.nrrd $INPUT_DIR/20170301_31_B5_Scope_1_C1_down_result_masked_warped.nrrd \
#     -rj $INPUT_DIR/20170301_31_B5_Scope_1_C1_down_result_masked_jac.nrrd


# Greedy on toy data
DATA_PATH=/nrs/scicompsoft/dingx/GAN_data/toy_data
mkdir -p $DATA_PATH/output
cd $DATA_PATH/output
for ((i=0; i<=999; i++)); do
    mkdir -p ./$i
    cd ./$i 
    $GREEDY_PATH/greedy -threads 16 -d 3 -m SSD \
        -i $DATA_PATH/sphere.nrrd $DATA_PATH/$i/warped.nrrd \
        -o ./phi.nrrd \
        -n 200x10
    $GREEDY_PATH/greedy -threads 16 -d 3 \
        -rf $DATA_PATH/sphere.nrrd \
        -rm $DATA_PATH/$i/warped.nrrd ./obj_warped.nrrd \
        -r ./phi.nrrd \
        -rj ./jac.nrrd
    cd ../
done 