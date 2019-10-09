# import pickle
import json
import nrrd
from model import SimpleUnet 
from modules import *
import torch.nn as nn 
import torch.optim as optim
import glob
import os 


model_path = '/nrs/scicompsoft/dingx/GAN_model/simpleunet_bf32_cc_sgd/'
# test data
with open(model_path+'data_loss.json', 'r') as f:
    saved_data = json.load(f)
test_list = saved_data['test_list']
# template
tmplt, head = nrrd.read('/nrs/scicompsoft/dingx/GAN_data/toy_data/sphere.nrrd')
tmplt = (tmplt-tmplt.mean()) / tmplt.std()  # normalize tmplt
# phi head used for writing
phi, phi_head = nrrd.read(model_path+'phi_sample.nrrd')
# checkpoint list
ckpt_list = glob.glob(model_path+'/model_ckpt_*.pt')

# Define the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SimpleUnet(in_channels=2, base_filters=32, out_channels=3).to(device)
criterion = cc_loss  # nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.95, 0.999))
input_sz = (64,64,64)

# Load checkpoints
network = ImgRegisterNetwork(model, criterion, optimizer, device)
for ckpt in ckpt_list:
    print("Test checkpoint {}".format(os.path.basename(ckpt)))
    for i in range(len(test_list)):
        img, head = nrrd.read(test_list[i])
        img = (img-img.mean()) / img.std()  # normalize img
        phi = network.test_model(ckpt, img, tmplt, input_sz)
        name = 'img{}_phi_'.format(i) + os.path.splitext(os.path.basename(ckpt))[0] + '.nrrd'
        nrrd.write(model_path+name, phi, phi_head)