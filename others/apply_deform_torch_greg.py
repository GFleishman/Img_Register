import nrrd
import torch
import torch.nn.functional as F
import numpy as np


file_path = "/nrs/scicompsoft/dingx/greg/"
move_img = file_path+'sphere.nrrd'
phi_img = file_path+'phiinv.nrrd'

# Moving image
move, head = nrrd.read(move_img)
move = np.expand_dims(move, axis=0) # [channel, x, y, z]
move = np.expand_dims(move, axis=0)  # [batch, channel, x, y, z]
move = torch.from_numpy(move).float()

# Deformable field
phi, head = nrrd.read(phi_img) # [x, y, z, channel]
phi = np.expand_dims(phi, axis=0)
print(phi.shape)
#phi = np.expand_dims(phi, axis=0) # [batch, channel, x, y, z]
#phi = np.transpose(phi, (0,2,3,4,1))  # [batch, x, y, z, channel]

# Add a base grid to deformable field
grid = phi.shape[1:-1]
#base_grid = np.array(np.meshgrid(*[range(x) for x in grid], indexing='ij'))
base_grid = np.array(np.meshgrid(*[np.linspace(-1, 1, x) for x in grid], indexing='ij'))
base_grid = np.ascontiguousarray(np.moveaxis(base_grid, 0, -1))
#base_grid = np.asarray(base_grid)  # [channel, y, x, z]
#base_grid = np.transpose(base_grid, (2,1,3,0))  # [x, y, z, channel], bug, check base_grid shape
#base_grid = np.expand_dims(base_grid, axis=0)  # [batch, x, y, z, channel]
phi *= 2./phi.shape[1]
phi += base_grid

# Scale to [-1,1]
#phi_min = phi.min(axis=(1,2,3), keepdims=True)
#phi_max = phi.max(axis=(1,2,3), keepdims=True)
#phi = (phi-phi_min) * 2 / (phi_max-phi_min) -1

phi = torch.from_numpy(phi).float()

warped = F.grid_sample(move, phi)
warped_img = warped.numpy()[0,0,:,:,:]

name = file_path+'out_greg.nrrd'
nrrd.write(name, warped_img)
