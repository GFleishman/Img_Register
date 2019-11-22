import nrrd
import numpy as np 
import torch
from torch.utils.data import Dataset
import glob
import nrrd
import os 


def random_crop(img, crop_point, crop_sz):
    """
    Randomly crop image
    Args:
        img: image in shape (x, y, z)
        crop_point: point (x, y, z) to start cropping
        crop_sz: cropping size in (x, y, z)
    Return:
        cropped image
    """
    return img[crop_point[0]:crop_point[0]+crop_sz[0], crop_point[1]:crop_point[1]+crop_sz[1], crop_point[2]:crop_point[2]+crop_sz[2]]


def flip_img(img, opt):
    """
    Flip image
    Args:
        img: image in shape (x, y, z)
        opt: flip option, 0-no flip, 1-x flip, 2-z flip, 3-both x and z flip
    Return:
        flipped img
    """
    if opt == 1:
        img = np.flip(img, axis=0)
    elif opt == 2:
        img = np.flip(img, axis=2)
    elif opt == 3:
        img = np.flip(img, axis=0)
        img = np.flip(img, axis=2)
    else:
        img = img
    return img


def rot_img(img, k):
    """
    Rotate image
    Args:
        img: image in shape (x, y, z)
        k: 1, 2, or 3 times of 90 degrees
    """
    if k:
        img = np.rot90(img, k, axes=(0,1))
    return img


class GenerateData(Dataset):
    """
    Generate training and validation data
    """
    def __init__(self, img_list, tmplt_name, crop_sz=(32,32,32)):
        """
        Args:
            img_list: a list of image names
            tmplt_name: template name
            crop_sz: random cropping size
        """
        self.img_list = img_list
        self.tmplt_name = tmplt_name
        self.crop_sz = crop_sz

    def __getitem__(self, idx):
        """
        Get specific data corresponding to the index
        Args:
            idx: data index
        Returns:
            tensor (img, tmplt)
        """
        # Get image and template
        img_name = self.img_list[idx]
        img, head = nrrd.read(img_name)
        img = np.float32(img)
        tmplt, head = nrrd.read(self.tmplt_name)
        tmplt = np.float32(tmplt)

        # Normalize image and template
        img = (img-img.mean()) / img.std()
        tmplt = (tmplt-tmplt.mean()) / tmplt.std()

        # Crop on image and template
        x = np.random.randint(0, img.shape[0]-self.crop_sz[0]+1)
        y = np.random.randint(0, img.shape[1]-self.crop_sz[1]+1)
        z = np.random.randint(0, img.shape[2]-self.crop_sz[2]+1)
        img = random_crop(img, (x,y,z), self.crop_sz)
        tmplt = random_crop(tmplt, (x,y,z), self.crop_sz)

        # Augmentation to image and template
        opt = np.random.randint(4)
        img = flip_img(img, opt)
        tmplt = flip_img(tmplt, opt)
        k = np.random.randint(4)
        img = rot_img(img, k)
        tmplt = rot_img(tmplt, k)

        # To tensor, shape (channel, x, y, z)
        img = np.expand_dims(img, axis=0)
        tmplt = np.expand_dims(tmplt, axis=0)
        img = np.concatenate((img, tmplt), axis=0)  # input has two channels, channel0: img, channel1: tmplt
        img = torch.from_numpy(img.copy()).float()
        tmplt = torch.from_numpy(tmplt.copy()).float()

        return [img, tmplt]

    def __len__(self):
        return len(self.img_list)


if __name__ == "__main__":

    data_path = '/nrs/scicompsoft/dingx/GAN_data/toy_data/'
    img_list = glob.glob(data_path+'*/warped.nrrd')
    tmplt_name = data_path+'sphere.nrrd'
    Data = GenerateData(img_list, tmplt_name)
    img, tmplt = Data.__getitem__(1)
    print(img.shape)
    print(tmplt.shape)

    # View part of the data
    img = img.numpy()
    tmplt = tmplt.numpy()
    img_ch0 = np.zeros((img.shape[1], img.shape[2], img.shape[3]), dtype=img.dtype)
    img_ch0 = img[0,:,:,:]
    img_ch1 = np.zeros((img.shape[1], img.shape[2], img.shape[3]), dtype=img.dtype)
    img_ch1 = img[1,:,:,:]
    tmplt_ch0 = np.zeros((tmplt.shape[1], tmplt.shape[2], tmplt.shape[3]), dtype=tmplt.dtype)
    tmplt_ch0 = tmplt[0,:,:,:]

    curr_path = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(curr_path+'/data_view'):
        os.mkdir(curr_path+'/data_view')
    nrrd.write(curr_path+'/data_view/img_ch0.nrrd', img_ch0)
    nrrd.write(curr_path+'/data_view/img_ch1.nrrd', img_ch1)
    nrrd.write(curr_path+'/data_view/tmplt_ch0.nrrd', tmplt_ch0)