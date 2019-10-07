import nrrd
import os

mask_file = '/nrs/scicompsoft/dingx/GAN_data/data_fly/JRC2018_lo_dilated_mask.nrrd'
img_file = '/nrs/scicompsoft/dingx/GAN_data/data_fly/data/20170301_31_B5_Scope_1_C1_down_result.nrrd'


mask_data, mask_head = nrrd.read(mask_file)
img_data, img_head = nrrd.read(img_file)

img_data = mask_data * img_data
name = os.path.splitext(img_file)[0] + '_masked.nrrd'

nrrd.write(name, img_data)