import sys
sys.path += ['/groups/scicompsoft/home/fleishmang/source/Img_Register/scripts_toy']

import numpy as np
import torch
from model import SimpleUnet
from modules import *
import torch.optim as optim
import nrrd

from data import GenerateData
from torch.utils.data import DataLoader


# get the validation data
fixed_path      = sys.argv[1]
moving_path       = sys.argv[2]
transform_path   = sys.argv[3]  # transform actually takes fixed image to moving image

moving, meta      = nrrd.read(moving_path)
fixed, meta       = nrrd.read(fixed_path)
transform, meta   = nrrd.read(transform_path)
transform         = np.moveaxis(transform * 2./63., -1, 0)
transform = np.array([transform[2], transform[1], transform[0]]) # torch seems to want [z, y, x] for displacements

moving_t     = torch.from_numpy(moving[None, None, ...]).float()
fixed_t      = torch.from_numpy( fixed[None, None, ...]).float()
transform_t  = torch.from_numpy(   transform[None, ...]).float()
transform_t  = transform_t.clone()


# initially, we just need the model and loss function
model = SimpleUnet(in_channels=2, base_filters=16, out_channels=3)
loss        = cc_loss


# evaluate basic tests
if sys.argv[4] in ['basic', 'both']:

    # test correlation coefficient code against numpy
    np_cc    = - np.corrcoef(fixed.flatten(), moving.flatten())[0, 1]
    my_cc, _ = loss(fixed_t, moving_t)
    print("CC TEST Numpy: ", np_cc, "\tours: ", my_cc, "\t", np.allclose(np_cc, my_cc), "\n")
   
    # test gradient code against numpy
    x            = np.gradient(transform[0, ...])
    y            = np.gradient(transform[1, ...])
    z            = np.gradient(transform[2, ...])
    np_grad      = np.array([ x[0], y[0], z[0], x[1], y[1], z[1], x[2], y[2], z[2] ])[None, ...]
    _, my_grad   = loss(fixed_t, moving_t, transform_t)
    print("GRAD TEST: ", np.allclose(np_grad, my_grad.numpy()), "\n")

    # test transform layer with a given transform
    warped = transform_layer(fixed_t, transform_t)
    print("KNOWN TRANSFORM TEST: ", np.allclose(moving, warped.detach().numpy().squeeze(), atol=1e-5), "\n")
    
    # test transform layer with initial value of model (identity)
    model_transform_t = model(torch.cat((moving_t, fixed_t), axis=1))
    warped = transform_layer(moving_t, model_transform_t)
    print("MODEL TRANSFORM TEST: ", np.allclose(moving, warped.detach().numpy().squeeze(), atol=1e-6), "\n")



# evaluate training based tests
if sys.argv[4] in ['train', 'both']:

    if len(sys.argv) == 5:
        moving_paths = [moving_path,]
    elif len(sys.argv) == 6:
        moving_paths = [moving_path, sys.argv[5]]
    else:
        moving_paths = [moving_path,] + sys.argv[5:]

    print(moving_paths)

    # construct the registration network
    device      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer   = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-3)
    network     = ImgRegisterNetwork(model, loss, optimizer, device)


    # load the data as DataLoader and train
    train_data     = GenerateData(moving_paths, fixed_path, (64, 64, 64))
    train_loader   = DataLoader(dataset=train_data, batch_size=1, shuffle=True)

    for i in range(200):
        train_loss     = network.train_model(train_loader, i)

