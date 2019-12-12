import torch
import torch.nn.functional as F 
import numpy as np 
import os


def transform_layer_position(img, phi):  # old version
    """
    Transformation layer. Phi is a position field.
    Args:
        img: images in shape [batch, channel, x, y, z], only first channel is the raw image
        phi: position field in shape [batch, channel, x, y, z]
    """
    phi = phi.permute(0,2,3,4,1)  # [batch, x, y, z, channel]

    # Scale to [-1,1]
    phi_min = torch.min(phi, dim=1, keepdim=True)
    phi_min = torch.min(phi_min.values, dim=2, keepdim=True)
    phi_min = torch.min(phi_min.values, dim=3, keepdim=True)
    phi_max = torch.max(phi, dim=1, keepdim=True)
    phi_max = torch.max(phi_max.values, dim=2, keepdim=True)
    phi_max = torch.max(phi_max.values, dim=3, keepdim=True)
    phi = (phi-phi_min.values) * 2 / (phi_max.values-phi_min.values) -1

    # Extract the first channel of img
    img = torch.split(img, 1, dim=1)  # split channel, the first channel is raw img
    
    # Apply deformable field
    warped = F.grid_sample(img[0], phi)

    return warped


def transform_layer_displacement(img, phi):  # old version
    """
    Transformation layer. Phi is a displacement field.
    Args:
        img: images in shape [batch, channel, x, y, z], only first channel is the raw image
        phi: displacement field in shape [batch, channel, x, y, z]
    """
    phi = phi.permute(0,2,3,4,1)  # [batch, x, y, z, channel]

    # Add a base grid to deformable field
    base_grid = torch.meshgrid(torch.linspace(0,phi.shape[1]-1,phi.shape[1]), torch.linspace(0,phi.shape[2]-1,phi.shape[2]), torch.linspace(0,phi.shape[3]-1,phi.shape[3]))
    base_grid = torch.stack(base_grid)  # [channel, x, y, z]
    base_grid = base_grid.permute(1,2,3,0)  # [x, y, z, channel]
    base_grid = base_grid.unsqueeze(0)  # [batch, x, y, z, channel]
    base_grid = base_grid.to(phi.dtype)
    base_grid = base_grid.to(phi.device)
    phi += base_grid 

    # Scale to [-1,1]
    phi_min = torch.min(phi, dim=1, keepdim=True)
    phi_min = torch.min(phi_min.values, dim=2, keepdim=True)
    phi_min = torch.min(phi_min.values, dim=3, keepdim=True)
    phi_max = torch.max(phi, dim=1, keepdim=True)
    phi_max = torch.max(phi_max.values, dim=2, keepdim=True)
    phi_max = torch.max(phi_max.values, dim=3, keepdim=True)
    phi = (phi-phi_min.values) * 2 / (phi_max.values-phi_min.values) -1

    # Extract the first channel of img
    img = torch.split(img, 1, dim=1)  # split channel, the first channel is raw img
    
    # Apply deformable field
    warped = F.grid_sample(img[0], phi)

    return warped


@torch.no_grad()
def generate_base_grid(phi):
    """
    Define a base grid
    Args:
        phi: displacement field in shape [batch, x, y, z, channel]
    Return a base grid with no gradient recorded
    """
    base_grid = torch.meshgrid(torch.linspace(-1,1,phi.shape[1]), torch.linspace(-1,1,phi.shape[2]), torch.linspace(-1,1,phi.shape[3]))
    base_grid = torch.stack(base_grid)  # [channel, x, y, z]
    base_grid = base_grid.permute(1,2,3,0)  # [x, y, z, channel]
    base_grid = base_grid.unsqueeze(0)  # [batch, x, y, z, channel]
    base_grid = base_grid.to(phi.dtype)

    # sz = phi.shape[1]
    # theta = torch.tensor([[[0,0,1,0],[0,1,0,0],[1,0,0,0]]], dtype=torch.float32)
    # base_grid = F.affine_grid(theta, torch.Size((1,1,sz,sz,sz)), align_corners=True)

    base_grid = base_grid.to(phi.device)
    return base_grid


def transform_layer(img, phi):
    """
    Transformation layer. Phi is a displacement field.
    Args:
        img: images in shape [batch, channel, x, y, z], only first channel is the raw image
        phi: displacement field in shape [batch, channel, x, y, z]
    """
    phi = phi.permute(0,2,3,4,1)  # [batch, x, y, z, channel]

    # Generate a base_grid
    base_grid = generate_base_grid(phi)
    # Scale phi and add to base grid
    # phi = phi*2 / phi.shape[1]
    phi += base_grid

    # Apply deformable field
    img = torch.split(img, 1, dim=1)  # split channel, the first channel is raw img
    warped = F.grid_sample(img[0], phi)

    return warped


def cc_loss(output, target, phi=None, lamda=None):
    '''
    Pearson correlation loss with smooth control of phi
    '''
    def calculate_gradient(phi):
        '''
        phi in shape [batch, channel, x, y, z]
        '''
        if phi is not None:
            grad_xc = 0.5 * (phi[:,:,2:,:,:] - phi[:,:,:-2,:,:])
            grad_xn = (phi[:,:,-1,:,:] - phi[:,:,-2,:,:]).unsqueeze(2)
            grad_x0 = (phi[:,:,1,:,:] - phi[:,:,0,:,:]).unsqueeze(2)
            grad_x = torch.cat((grad_x0,grad_xc,grad_xn), dim=2)

            grad_yc = 0.5 * (phi[:,:,:,2:,:] - phi[:,:,:,:-2,:])
            grad_yn = (phi[:,:,:,-1,:] - phi[:,:,:,-2,:]).unsqueeze(3)
            grad_y0 = (phi[:,:,:,1,:] - phi[:,:,:,0,:]).unsqueeze(3)
            grad_y = torch.cat((grad_y0,grad_yc,grad_yn), dim=3)

            grad_zc = 0.5 * (phi[:,:,:,:,2:] - phi[:,:,:,:,:-2])
            grad_zn = (phi[:,:,:,:,-1] - phi[:,:,:,:,-2]).unsqueeze(4)
            grad_z0 = (phi[:,:,:,:,1] - phi[:,:,:,:,0]).unsqueeze(4)
            grad_z = torch.cat((grad_z0,grad_zc,grad_zn), dim=4)

            # grad_phi = torch.sum(torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2), dim=(1,2,3,4))
            # grad_phi = torch.mean(grad_phi)
            grad_xyz = torch.cat([grad_x, grad_y, grad_z], dim=1)
            grad_phi = torch.sqrt(torch.sum(grad_xyz**2, dim=(1,2,3,4)))
            grad_phi = torch.mean(grad_phi)
            return grad_phi         
        else:
            pass
    
    x = output - torch.mean(output, dim=(2,3,4), keepdim=True)
    y = target - torch.mean(target, dim=(2,3,4), keepdim=True)
    cc = torch.sum(x * y, dim=(1,2,3,4)) / (torch.sqrt(torch.sum(x ** 2, dim=(1,2,3,4))) * torch.sqrt(torch.sum(y ** 2, dim=(1,2,3,4))))
    cc = torch.mean(cc)
    if phi is not None:
        smooth = calculate_gradient(phi)
        if lamda is None:
            lamda = torch.tensor([1.0])
        lamda = lamda.to(phi.dtype)
        lamda = lamda.to(phi.device)
        loss = -cc + lamda * smooth
    else:
        loss = -cc
    return loss


class ImgRegisterNetwork():
    """
    Image registeration network class that wraps model related functions (e.g., training, evaluation, etc)
    """
    def __init__(self, model, criterion, optimizer, device):
        """
        Args:
            model: a deep neural network model (sent to device already)
            criterion: loss function
            optimizer: training optimizer
            device: training device
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device


    def train_model(self, data):
        """
        Train the model
        Args:
            data: training dataset generated by DataLoader
        Return batch-wise training loss
        """
        self.model.train()
        training_loss = 0

        for batch, [img, tmplt] in enumerate(data):
            img = img.to(self.device)
            tmplt = tmplt.to(self.device)

            # Forward
            phi = self.model(img)
            # Apply transformation layer
            warped = transform_layer_displacement(img, phi)
            # Calculate loss
            loss = self.criterion(warped, tmplt, phi)
            training_loss += loss.item()
            # Zero the parameter gradients
            self.optimizer.zero_grad()                    
            # Backward
            loss.backward()
            # Update weights
            self.optimizer.step()

        print("Batch-wise training loss for current epoch is {}".format(training_loss/(batch+1)))
        return training_loss/(batch+1)


    def eval_model(self, data):
        """
        Evaluate the model
        Args:
            data: evaluation dataset generated by DataLoader
        Return batch-wise evaluation loss
        """
        self.model.eval()
        eval_loss = 0

        for batch, [img, tmplt] in enumerate(data):
            with torch.no_grad():  # Disable gradient computation
                img = img.to(self.device)
                tmplt = tmplt.to(self.device)
                phi = self.model(img)
                warped = transform_layer_displacement(img, phi)
                loss = self.criterion(warped, tmplt, phi)
                eval_loss += loss.item()

        print("Batch-wise evaluation loss for current epoch is {}".format(eval_loss/(batch+1)))
        return eval_loss/(batch+1)


    def save_model(self, path, epoch, entire=False):
        """
        Save the model to disk
        Args:
            path: directory to save the model
            epoch: epoch that model is saved
            entire: if save the entire model rather than just save the state_dict
        """
        if not os.path.exists(path):
            os.mkdir(path)
        if entire:
            torch.save(self.model, path+"/whole_model_epoch_{}.pt".format(epoch))
        else:
            # torch.save(self.model.state_dict(), path+"/model_ckpt_{}.pt".format(epoch))
            torch.save({'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'criterion': self.criterion},
                        path+"/model_ckpt_{}.pt".format(epoch))
    

    def test_model(self, checkpoint, img, tmplt, input_sz):
        """
        Test the model on new data
        Args:
            checkpoint: saved checkpoint
            img: testing data (moving image)
            tmplt: template (fixed image)
            input_sz: network input size in (x,y,z) (network input is [batch, channel, x, y, z])
        """
        assert img.shape == tmplt.shape, "moving image doesn't match fixed template shape!"

        ckpt = torch.load(checkpoint)
        # self.model.load_state_dict(ckpt)
        self.model.load_state_dict(ckpt['model_state_dict'])

        self.model.eval()

        phi = np.zeros((3, img.shape[0], img.shape[1], img.shape[2]), dtype=img.dtype)
        warped  = np.zeros(img.shape, dtype=img.dtype)
        for row in range(0, img.shape[0], input_sz[0]):
            for col in range(0, img.shape[1], input_sz[1]):
                for vol in range(0, img.shape[2], input_sz[2]):
                    # Generate 
                    patch_img = np.zeros((1, 2, input_sz[0], input_sz[1], input_sz[2]), dtype=img.dtype)
                    patch_img[0,0,:,:,:] = img[row:row+input_sz[0], col:col+input_sz[1], vol:vol+input_sz[2]]
                    patch_img[0,1,:,:,:] = tmplt[row:row+input_sz[0], col:col+input_sz[1], vol:vol+input_sz[2]]
                    patch_img = torch.from_numpy(patch_img).float()
                    patch_img = patch_img.to(self.device)
                    # Apply model
                    patch_phi = self.model(patch_img)
                    patch_warped = transform_layer_displacement(patch_img, patch_phi)

                    patch_phi = patch_phi.cpu()
                    patch_phi = patch_phi.detach().numpy()
                    phi[:, row:row+input_sz[0], col:col+input_sz[1], vol:vol+input_sz[2]] = patch_phi[0,:,:,:,:]
                    patch_warped = patch_warped.cpu()
                    patch_warped = patch_warped.detach().numpy()
                    warped[row:row+input_sz[0], col:col+input_sz[1], vol:vol+input_sz[2]] = patch_warped
        return phi, warped
