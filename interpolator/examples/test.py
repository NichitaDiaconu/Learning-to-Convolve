"""Various intepolation schemes for PyTorch"""
import os
import sys
import time

import numpy as np
import torch

from matplotlib import pyplot as plt
from skimage.io import imread

from core import Interpolator
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
image = imread('./owl.jpg').transpose(2,0,1)[np.newaxis,...]/255.

# Let's blur the owl
## Define a grid of sample locations
new_shape = np.asarray(image.shape[2:]) // 2
I, J = np.meshgrid(np.linspace(1, image.shape[2], num=new_shape[0]),
                   np.linspace(1, image.shape[3], num=new_shape[1]),
                   indexing='ij')
samples = np.stack((I,J), 0)

# Convert to pytorch tensors
image = torch.FloatTensor(image).type(dtype)
samples = torch.FloatTensor(samples).type(dtype)

# Create an interpolator
#itp = Interpolator(BilinearKernel(), samples)
print('Image shape: {}'.format(image.shape))
print('Sample shape: {}'.format(samples.size()))
#itp = Interpolator(BilinearKernel(), samples)
itp = Interpolator(GaussianKernel(5, 1), samples)

# Run the interpolator
start = time.time()
new_image = itp.forward(image)
duration = time.time() - start
print("Duration: {}".format(duration))

# View
new_shape = np.append(3,new_shape)
new_shape = np.append(1,new_shape)
new_image = torch.reshape(new_image,tuple(new_shape)).cpu().numpy()
view(new_image[0,...])
