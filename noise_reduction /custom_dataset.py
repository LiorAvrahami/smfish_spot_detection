import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as func
import torchvision.models as models
import torchvision.transforms as tfms
from PIL import Image
from skimage.draw import random_shapes


class CustomDataset(Dataset):
    def __init__(self, image_list, image_labels=None):
        self.images = [image_list[i // 3][0][:, :, :, i % 3] for i in range(len(image_list) * 3)]  # gives a list
        self.clean_images = [image_list[i // 3][1][:, :, :, i % 3] for i in range(len(image_list) * 3)]

    def __len__(self):
        return len(self.images)

    ### convert image to suitable format for pytorch, but tensorflow. We assume our images are 100X100X10X3 to be converted to 100X100X10X1

    def __getitem__(self, idx):
        input_img = torch.tensor(self.images[idx].copy()).float()
        input_img = input_img.permute(2, 1, 0)
        targ_img = torch.tensor(self.clean_images[idx].copy()).float()  # We want to predict the clean image of only the spots
        targ_img = targ_img.permute(2, 1, 0)
        return input_img, targ_img
