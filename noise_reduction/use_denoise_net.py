import csv
import itertools
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import torch


def clean_image(image: npt.NDArray, denoise_net) -> npt.NDArray:
    image = torch.tensor(np.copy(image))  # type: ignore
    image = image.permute(3, 2, 1, 0)  # get ch,z,x,y # type: ignore
    image_real = image.unsqueeze(1)  # type: ignore
    pred = denoise_net(image_real.float()).detach()
    pred = pred[:, 0, :, :, :]
    pred = pred.permute(3, 2, 1, 0)
    return pred.numpy()
