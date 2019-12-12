import nrrd
import torch
import torch.nn.functional as F
import numpy as np


file_path = "/nrs/scicompsoft/dingx/greg/"

# Moving image
move, head = nrrd.read(file_path+"sphere.nrrd")
move = np.expand_dims(move, axis=0) # [channel, x, y, z]
move = np.expand_dims(move, axis=0)  # [batch, channel, x, y, z]
move = torch.from_numpy(move).float()

# Deformable field
phi, phi_head = nrrd.read(file_path+"phiinv.nrrd") # [x, y, z, channel]
phi = np.expand_dims(phi, axis=0) # [batch, x, y, z, channel]

# # Add a base grid to deformable field
# base_grid = np.meshgrid(np.linspace(0,phi.shape[1]-1,phi.shape[1]), np.linspace(0,phi.shape[2]-1,phi.shape[2]), np.linspace(0,phi.shape[3]-1,phi.shape[3]), indexing='ij')
# base_grid = np.asarray(base_grid)  # [channel, x, y, z]
# base_grid = np.transpose(base_grid, (1,2,3,0))  # [x, y, z, channel]
# base_grid = np.expand_dims(base_grid, axis=0)  # [batch, x, y, z, channel]
# phi += base_grid

# # Scale to [-1,1]
# phi_min = phi.min(axis=(1,2,3), keepdims=True)
# phi_max = phi.max(axis=(1,2,3), keepdims=True)
# phi = (phi-phi_min) * 2 / (phi_max-phi_min) -1


sz = phi.shape[1]
theta = torch.tensor([[[0,0,1,0],[0,1,0,0],[1,0,0,0]]], dtype=torch.float32)
base_grid = F.affine_grid(theta, torch.Size((1,1,sz,sz,sz)), align_corners=True)
# base_out = base_grid.numpy()[0,:,:,:,:]
# nrrd.write(file_path+'base_grid_torch.nrrd', base_out, phi_head)


# base_grid = np.meshgrid(np.linspace(-1,1,phi.shape[1]), np.linspace(-1,1,phi.shape[2]), np.linspace(-1,1,phi.shape[3]), indexing='ij')
# base_grid = np.asarray(base_grid)
# base_grid = np.transpose(base_grid, (1,2,3,0))
# nrrd.write(file_path+'base_grid_numpy.nrrd', base_grid, phi_head)
# base_grid = np.expand_dims(base_grid, axis=0)


phi = phi*2/phi.shape[1]
phi = torch.from_numpy(phi).float()
phi += base_grid



warped = F.grid_sample(move, phi)
warped_img = warped.numpy()[0,0,:,:,:]

nrrd.write(file_path+'out_torch.nrrd', warped_img)