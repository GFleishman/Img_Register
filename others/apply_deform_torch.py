import nrrd
import torch
import torch.nn.functional as F
import numpy as np


move_img = "/nrs/scicompsoft/dingx/GAN_data/toy_data/sphere.nrrd"
phi_img = "/nrs/scicompsoft/dingx/GAN_data/toy_data/playground/warp.nrrd"

# Moving image
move, head = nrrd.read(move_img)
move = np.expand_dims(move, axis=0) # [channel, x, y, z]
move = np.expand_dims(move, axis=0)  # [batch, channel, x, y, z]
move = torch.from_numpy(move).float()

# Deformable field
phi, head = nrrd.read(phi_img) # [channel, x, y, z]
phi = np.expand_dims(phi, axis=0) # [batch, channel, x, y, z]

phi = np.transpose(phi, (0,2,3,4,1))  # [batch, x, y, z, channel]

# Add a base grid to deformable field
base_grid = np.meshgrid(np.linspace(0,phi.shape[1]-1,phi.shape[1]), np.linspace(0,phi.shape[2]-1,phi.shape[2]), np.linspace(0,phi.shape[3]-1,phi.shape[3]))
base_grid = np.asarray(base_grid)  # [channel, y, x, z]
base_grid = np.transpose(base_grid, (2,1,3,0))  # [x, y, z, channel]
base_grid = np.expand_dims(base_grid, axis=0)  # [batch, x, y, z, channel]
phi += base_grid

# Scale to [-1,1]
phi_min = phi.min(axis=(1,2,3), keepdims=True)
phi_max = phi.max(axis=(1,2,3), keepdims=True)
phi = (phi-phi_min) * 2 / (phi_max-phi_min) -1

phi = torch.from_numpy(phi).float()

warped = F.grid_sample(move, phi)
warped_img = warped.numpy()[0,0,:,:,:]

name = "/nrs/scicompsoft/dingx/GAN_data/toy_data/playground/sphere_warped_torch.nrrd"
nrrd.write(name, warped_img)