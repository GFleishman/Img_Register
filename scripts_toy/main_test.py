# import pickle
import json
import nrrd
from model import SimpleUnet 
from modules import *
import torch.nn as nn 
import torch.optim as optim
import glob
import os 


model_path = '/nrs/scicompsoft/dingx/GAN_model/simpleunet_input2channel_insz64/'
# test data
with open(model_path+'data_loss.json', 'r') as f:
    saved_data = json.load(f)
test_list = saved_data['test_list']
# template
tmplt, head = nrrd.read('/nrs/scicompsoft/dingx/GAN_data/toy_data/sphere.nrrd')
# phi head used for writing
phi, phi_head = nrrd.read('/nrs/scicompsoft/dingx/GAN_model/simpleunet_input2channel_insz32/img0_phi_model_ckpt_100000.nrrd')
# checkpoint list
ckpt_list = glob.glob(model_path+'/model_ckpt_*.pt')

# Define the model
model = SimpleUnet(in_channels=2, base_filters=32, out_channels=3)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.95, 0.999))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_sz = (64,64,64)

# Load checkpoints
network = ImgRegisterNetwork(model, criterion, optimizer, device)
for ckpt in ckpt_list:
    print("Test checkpoint {}".format(os.path.basename(ckpt)))
    for i in range(len(test_list)):
        img, head = nrrd.read(test_list[i])
        phi = network.test_model(ckpt, img, tmplt, input_sz)
        name = 'img{}_phi_'.format(i) + os.path.splitext(os.path.basename(ckpt))[0] + '.nrrd'
        nrrd.write(model_path+name, phi, phi_head)