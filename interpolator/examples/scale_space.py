'''Various intepolation schemes for PyTorch'''
import os
import sys
import time

import numpy as np
import torch

from matplotlib import pyplot as plt
from skimage.io import imread
from torch import nn

from core import self
from kernels import BilinearKernel, GaussianKernel

dtype = torch.FloatTensor
dtype_long = torch.LongTensor


def view(image):
    if image.shape[0] == 3:
        image = image.transpose(1,2,0)
    else:
        image = image[0,...]
    plt.figure(1, figsize=(12,8))
    plt.imshow(image)
    plt.show()


# Source: https://www.flickr.com/photos/48509939@N07/5927758528/in/photostream/
image = imread('owl.jpg').transpose(2,0,1)[np.newaxis,...]/255.

# Convert to pytorch tensors
image = torch.FloatTensor(image).type(dtype)
# Create a continuous scale-space
scale_space = []
nlevels = 10
sigma = .5
scale = np.power(np.power(0.5, 1/3), np.arange(nlevels+1))

H, W = image.shape[2], image.shape[3]
for level in range(nlevels):
    h_old = int(np.floor(H * scale[level]))-1
    w_old = int(np.floor(W * scale[level]))-1
    h = int(np.floor(H * scale[level+1]))
    w = int(np.floor(W * scale[level+1]))

    I, J = np.meshgrid(np.linspace(0, h_old, num=h),
                       np.linspace(0, w_old, num=w), indexing='ij')
    samples = np.stack((I,J), 0)
    samples = torch.FloatTensor(samples).type(dtype)

    scale_space.append(self(GaussianKernel(5, sigma), samples))

scale_space = nn.Sequential(*scale_space)

image = image.cuda()
scale_space = scale_space.cuda()
# Run the interpolator
start = time.time()
image = scale_space.forward(image)
duration = time.time() - start
print("Duration: {}".format(duration / 10.))

# View
image = image.cpu().numpy()
view(image[0,...])
