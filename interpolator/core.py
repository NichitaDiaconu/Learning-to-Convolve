"""Various interpolation schemes for PyTorch"""
import torch
from torch import nn


from constants import BATCH_SIZE


class Interpolator:
    """Interpolator class"""

    def __init__(self, kernel, device, image_size, padding_sz=1, padding_fnc=None):
        """

        :param kernel:
        :param device:
        :param batch_size: i.e 32
        :param image_size: i.e. (28, 28)
        :param padding_sz:
        :param padding_fnc:
        """
        # We can only pad by 1 if we want zero padding
        # We don't support any other type of padding at the moment
        self.padding_sz = padding_sz
        if padding_fnc is None:
            self.padding_fnc = nn.ZeroPad2d((self.padding_sz,
                                             self.padding_sz,
                                             self.padding_sz,
                                             self.padding_sz))
        else:
            self.padding_fnc = padding_fnc

        self.kernel = kernel

        # prpecompute some indices
        diffi = (torch.arange(self.kernel.kh) - ((self.kernel.kh - 1) // 2))
        diffj = (torch.arange(self.kernel.kw) - ((self.kernel.kw - 1) // 2))
        diffi = diffi.unsqueeze(-1).repeat(1, self.kernel.kw)
        diffj = diffj.repeat(self.kernel.kh, 1)
        self.diff = torch.stack([diffi, diffj], 0).to(device)

        # precompute some indices
        batch_shape = (*image_size, self.kernel.kh, self.kernel.kw)

        self.batch_idx = torch.ones((BATCH_SIZE, *batch_shape), dtype=torch.long, device=device) * \
                         torch.arange(0, BATCH_SIZE, device=device). \
                             unsqueeze(-1). \
                             unsqueeze(-1). \
                             unsqueeze(-1). \
                             unsqueeze(-1)

    def _get_patches_indices(self, grid):
        """
        Get indices of the closest pixel for each grid point. We call these centers
        Then get the indices for the kernels around these centers. We call centerd_patches

        :param grid: tensor with a batch of grid positions, one for each image
            shape = [batch_sz, 2, image_h, image_w]
        :param kernel: object of type Kernel. Must implement method get_weights
        :return: coordinate tensor of each kernel around each point in grid
            shape = [batch_sz, 2, image_h, image_w, kernel_h, kernel_w]
        """
        # Get closed pixel coordinates
        if 0 == (self.kernel.kh % 2):
            # If even kernel_size then use top left pixel as pixel_center
            centers = torch.floor(grid).to(torch.int64)
        else:
            # If odd kernel_size then use nearest pixel as center
            centers = torch.round(grid).to(torch.int64)

        # Get square patch - construct position differences and add to points
        # diffi = (torch.arange(self.kernel.kh) - ((self.kernel.kh - 1) // 2)).to(centers.device)
        # diffj = (torch.arange(self.kernel.kw) - ((self.kernel.kw - 1) // 2)).to(centers.device)
        # diffi = diffi.unsqueeze(-1).repeat(1, self.kernel.kw)
        # diffj = diffj.repeat(self.kernel.kh, 1)
        # diff = torch.stack([diffi, diffj], 0)
        return centers.unsqueeze(-1).unsqueeze(-1) + self.diff.unsqueeze(1).unsqueeze(1)

    def _crop_pad_patches_indices(self, size, patches_indices):
        """
        Truncate values in :param patches_indices: to [0, img_h+2*pad_size-1] and [0, +2*pad_size-1],
        depending on the axis of each index

        :param size: size of unapdded image
        :param patches_indices: indices at which we want so sample the image
            shape = [batch_sz, 2, image_h, image_w, kernel_h, kernel_w]
        :return: cropped indices where we will sample the image
            shape = [batch_sz, 2, image_h, image_w, kernel_h, kernel_w]
        """
        # Pad to [-pad_size, img_h + pad_size - 1]
        ones = torch.ones(1, dtype=patches_indices.dtype, device=patches_indices.device)
        pad_size = torch.tensor(self.padding_sz, dtype=patches_indices.dtype, device=patches_indices.device)
        # size = torch.tensor(size, dtype=patches_indices.dtype, device=patches_indices.device)
        size = size.clone().type(patches_indices.dtype).to(patches_indices.device)
        patches_indices = torch.max(patches_indices, -pad_size)
        patches_indices[:, 0, ...] = torch.min(patches_indices[:, 0, ...], size[-2] + pad_size - ones)
        patches_indices[:, 1, ...] = torch.min(patches_indices[:, 1, ...], size[-1] + pad_size - ones)

        # Now normalize indices to [0, img_h + 2*pad_size - 1]
        patches_indices = patches_indices + pad_size
        return patches_indices

    def _get_offsets(self, grid, patches_indices):
        """
        Computes offset between each point in :param grid: and each point in the kernel of that grid point.
        :param grid: the grid were we want to sample the image
            shape = [batch_sz, 2, image_h, image_w]
        :param patches_indices: a tensor with kernels for each point in the grid
            shape = [batch_sz, 2, image_h, image_w, kernel_h, kernel_w]
        :return: offsets: the tensor of offsets between each point in the grid and each point in its kernel
            shape = [batch_sz, 2, image_h, image_w, kernel_h, kernel_w]
        """
        offsets = patches_indices.to(grid.dtype) - grid.unsqueeze(-1).unsqueeze(-1)
        return offsets

    def _get_patches_weights(self, offsets):
        return self.kernel.get_weight(offsets)

    def grid_sample(self, images, grid):
        """
        Interpolate values at position :param grid: in :param images: using
        the interpolation method :param kernel:.
        For each image in :param images: one different grid is specified, therefore, one can
        transform a batch of images with different affine transformation for each image.
        padding of zero is used

        :param images: tensor with a batch of images.
            shape = [batch, channels, image_W, image_H]
        :param grid: tensor with a batch of grid positions, one for each image
            shape = [batch_sz, 2, image_w, image_h]
        :return: the resulting interpolated tensor
            shape = [batch_sz, image_channels, image_h, image_w]
        """
        input_size = torch.tensor(images.shape[-2:], dtype=torch.float, device=images.device)

        # Get centerd_patches around each grid point
        centered_patches_indices = self._get_patches_indices(grid)
        # Crop indices outside border and put them inside the border at the closest element
        cropped_padded_centered_patches_indices = self._crop_pad_patches_indices(input_size,
                                                                      centered_patches_indices)
        # Compute offset between the grid points and each point in its respective kernel
        offsets = self._get_offsets(grid, centered_patches_indices)
        # Compute weights of the kernel for each of the offsets
        weights = self._get_patches_weights(offsets)

        # Pad the images
        padded_images = self.padding_fnc(images)

        # Extract values from the images, at the position of the weights
        # batch_size = padded_images.shape[0]
        # batch_shape = cropped_padded_centered_patches_indices[0, 0, ...].shape
        #
        # batch_idx = torch.ones((batch_size, *batch_shape), dtype=torch.long, device=padded_images.device) * \
        #             torch.arange(0, batch_size, device=padded_images.device). \
        #                 unsqueeze(-1). \
        #                 unsqueeze(-1). \
        #                 unsqueeze(-1). \
        #                 unsqueeze(-1)
        samples = padded_images[self.batch_idx[:images.shape[0], :images.shape[-2], :images.shape[-1]],
                                ...,
                                cropped_padded_centered_patches_indices[:, 0, ...],
                                cropped_padded_centered_patches_indices[:, 1, ...]]
        batch_dim = 0
        channels_dims = list(range(5, len(samples.shape)))
        image_kernel_dims = list(range(1, 5))
        samples_dims = (batch_dim, *channels_dims, *image_kernel_dims)
        samples = samples.permute(samples_dims)

        # Combine samples from images with the weights associated to them
        # prepare weights to broadcast on the color channel of images: weights.unsqueeze(1)
        # the resulting new values are summed over kernel dimensions
        for _ in range(len(channels_dims)):
            weights = weights.unsqueeze(1)
        new_values = torch.sum(torch.sum(samples * weights, -1), -1)
        return new_values
