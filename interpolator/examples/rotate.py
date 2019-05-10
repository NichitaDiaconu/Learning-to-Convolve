import time
import numpy as np
import torch
from skimage.io import imread
from transform_tensor_batch import TransformTensorBatch
from interpolator.kernels import BilinearKernel, GaussianKernel
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Source: https://www.flickr.com/photos/48509939@N07/5927758528/in/photostream/
image = imread('./owl.jpg').transpose(2, 0, 1)[np.newaxis, ...] / 255.
#for small patch
#images = image[:, :, 55:58, 55:58]
#for fixed values
#images[:, :] = torch.arange(0, 0.9, 0.1).view(3,3)

images = np.vstack((image, image, image, image))

# Convert to pytorch tensors
images = torch.tensor(images, device=DEVICE, dtype=torch.float)

#kernel = GaussianKernel(3, 1)
kernel = BilinearKernel()
transformer = TransformTensorBatch(kernel, images.shape[-2:], DEVICE)
rot_thetas = torch.tensor([np.pi/2, 3*np.pi/2, np.pi/3, 2*np.pi/3], device=DEVICE)

start = time.time()
images = transformer.rotate_and_scale_images(images, rot_thetas, method='interpolator')
duration = time.time() - start
print("Duration1: {}".format(duration))

# images2 = images.clone()
# images2 = images2.cpu().cuda()
# print(images.device)
images.cpu().cuda()
start = time.time()
images = transformer.rotate_and_scale_images(images, rot_thetas, method='interpolator')
duration = time.time() - start
print("Duration2: {}".format(duration))

#new_images.cpu()
start = time.time()
images = transformer.rotate_and_scale_images(images, rot_thetas, method='interpolator')
duration = time.time() - start
print("Duration3: {}".format(duration))

# plt.figure(); plt.imshow(np.transpose(images[0],(1,2,0))); plt.title('0'); plt.savefig('i_0.png')
# plt.figure(); plt.imshow(np.transpose(images[1],(1,2,0))); plt.title('1'); plt.savefig('i_1.png')
# plt.figure(); plt.imshow(np.transpose(images[2],(1,2,0))); plt.title('2'); plt.savefig('i_2.png')
# plt.figure(); plt.imshow(np.transpose(images[3],(1,2,0))); plt.title('3'); plt.savefig('i_3.png')

# plt.figure(); plt.imshow(images[0].permute(1,2,0)); plt.title('original_image')
# plt.figure(); plt.imshow(np.transpose(new_images[0],(1,2,0))); plt.title('0_0'); plt.savefig('t_0.png')
# plt.figure(); plt.imshow(np.transpose(new_images[1],(1,2,0))); plt.title('0_1'); plt.savefig('t_1.png')
# plt.figure(); plt.imshow(np.transpose(new_images[2],(1,2,0))); plt.title('0_2'); plt.savefig('t_2.png')
# plt.figure(); plt.imshow(np.transpose(new_images[3],(1,2,0))); plt.title('0_3'); plt.savefig('t_3.png')
# plt.show()

# plt.figure(); plt.imshow(np.transpose(new_new_images[0],(1,2,0))); plt.title('1_0'); plt.savefig('t_0.png')
# plt.figure(); plt.imshow(np.transpose(new_new_images[1],(1,2,0))); plt.title('1_1'); plt.savefig('t_1.png')
# plt.figure(); plt.imshow(np.transpose(new_new_images[2],(1,2,0))); plt.title('1_2'); plt.savefig('t_2.png')
# plt.figure(); plt.imshow(np.transpose(new_new_images[3],(1,2,0))); plt.title('1_3'); plt.savefig('t_3.png')
# plt.show()