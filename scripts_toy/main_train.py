import glob
import torch
from torch.utils.data import DataLoader
import torch.nn as nn 
import torch.optim as optim
from data import GenerateData
from model import SimpleUnet
from modules import *
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt 
# import pickle
import json
import os 


data_path = '/nrs/scicompsoft/dingx/GAN_data/toy_data'
img_list = glob.glob(data_path+'/*/warped.nrrd')
train_list = img_list[:-10]
eval_list = img_list[-10:-2]
test_list = img_list[-2:]
tmplt_name = data_path+'/sphere.nrrd'

save_path = '/nrs/scicompsoft/dingx/GAN_model/simpleunet_input2channel_insz64_cc_sgd'
if not os.path.exists(save_path):
    os.mkdir(save_path)
# use tensorboard
writer = SummaryWriter(save_path+"/log")

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model
model = SimpleUnet(in_channels=2, base_filters=32, out_channels=3).to(device)
print("Print model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, '\t', model.state_dict()[param_tensor].size())

# criterion
criterion = cc_loss # nn.MSELoss()
# optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.00005, nesterov=True)

network = ImgRegisterNetwork(model, criterion, optimizer, device)
batch_sz = 4
crop_sz=(64,64,64)

total_epoch = 10000
train_loss_total = []
eval_loss_total = []
train_loss_tb = 0
save_iter = 100

for epoch in range(total_epoch):
    # generate data for each epoch
    train_data = GenerateData(train_list, tmplt_name, crop_sz)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_sz, shuffle=True)

    train_loss = network.train_model(train_loader)
    train_loss_tb += train_loss

    if epoch%save_iter == save_iter-1:
        writer.add_scalar('training loss', train_loss_tb/save_iter, epoch)  # average training loss every save_iter
        train_loss_total.append(train_loss_tb/save_iter)
        train_loss_tb = 0
        print('Evaluating at {} epochs...'.format(epoch+1))
        eval_data = GenerateData(eval_list, tmplt_name, crop_sz)
        eval_loader = DataLoader(dataset=eval_data, batch_size=batch_sz)
        eval_loss = network.eval_model(eval_loader)
        writer.add_scalar('eval loss', eval_loss, epoch)
        eval_loss_total.append(eval_loss)

    if epoch%save_iter == save_iter-1:
        network.save_model(save_path, epoch+1)

save_dict = {'train_list': train_list, 'eval_list': eval_list, 'test_list': test_list, 'train_loss_total': train_loss_total, 'eval_loss_total': eval_loss_total}

# with open(save_path+'/data_loss.pkl', 'wb') as f:
#     pickle.dump(save_dict, f)
with open(save_path+'/data_loss.json', 'w') as f:
    json.dump(save_dict, f)

# plot loss
plt.figure()
plt.subplot(2,1,1)
plt.plot(range(len(train_loss_total)), train_loss_total)
plt.xlabel('epochs(x100)')
plt.ylabel('loss')
plt.title('Train loss')
plt.subplot(2,1,2)
plt.plot(range(len(eval_loss_total)), eval_loss_total)
plt.xlabel('epochs(x100)')
plt.ylabel('loss')
plt.title('Eval loss')

plt.tight_layout()
plt.savefig(save_path+'/loss.pdf')
plt.show()