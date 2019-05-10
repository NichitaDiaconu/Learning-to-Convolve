import torch
import scipy.ndimage.interpolation
import numpy as np

from constants import BATCH_SIZE
from interpolator.core import Interpolator


class TransformTensorBatch:

    def __init__(self, kernel, image_size, device, group_sz, use_scipy_order2=False):
        """

        :param kernel:
        :param device:
        :param image_size: expected height and width of images
        """
        self.use_scipy_order2 = use_scipy_order2
        self.image_size = image_size
        assert len(image_size) == 2
        self.interpolator = Interpolator(kernel, image_size=image_size, device=device)

        # If we have fixed size inputs, we can precompute some indices:
        # create meshgrid
        I, J = torch.meshgrid((torch.arange(0, image_size[-2], dtype=torch.float32),
                               torch.arange(0, image_size[-1], dtype=torch.float32)))
        # TODO this is very slow, check how and if worth it
        I = I.to(device)
        J = J.to(device)
        self.IJ = torch.stack((I, J), 0)

        self.group_range = torch.arange(0, group_sz, dtype=torch.long, device=device)
        self.batch_group_indices = torch.ones(group_sz, dtype=torch.long, device=device)[None, :] * torch.arange(0,
                                                                                                                 BATCH_SIZE,
                                                                                                                 device=device)[
                                                                                                    :, None]
        self.group_group_indices = torch.ones(group_sz, dtype=torch.long, device=device)[None, :] * torch.arange(0,
                                                                                                                 group_sz,
                                                                                                                 device=device)[
                                                                                                    :, None]

    def batch_roll(self, tensor, shift, axis, group_sz, dim=0):
        """
        given a tensor, across the 1st dimension, each tensor is rolled along axis dimension, shift positions
        each tensor on the 1st dimension (i.e. tensor[i])
        is rolled a specific nr of positions (is rolled shift[i] positions)
        and returns a new tensor with the result.
        len(shift) == tensor.shape[0]

        :param tensor: tensor
            shape = [batch_sz, ...]
        :param shift: array[int] how many positions to roll each tensor
        :param axis: int along which axis to roll all tensors
        :return: tensor with each tensor on 1dt dimension rolled shift positions along axis
            shape = [batch_sz, ...]
        """
        shifted_indices = self.group_range[None, :] - shift.long()[:, None]
        if dim == 0:
            gathered_tensor = tensor[self.batch_group_indices.view(-1), :, shifted_indices.view(-1)]
        elif dim == 1 and axis == 3:
            gathered_tensor = tensor[:, self.group_group_indices.view(-1), :, shifted_indices.view(-1)]
        elif dim == 1 and axis == 4:
            gathered_tensor = tensor[:, self.group_group_indices.view(-1), :, :, shifted_indices.view(-1)]
        indices_list = [i for i in range(len(tensor.shape))]
        permuted_axis = indices_list.copy()
        permuted_axis.pop(axis)
        permuted_axis.insert(0, axis)
        permuted_axis.pop(dim + 1)
        permuted_axis.insert(0, dim)

        inverse_permutation = np.array(indices_list)
        inverse_permutation[permuted_axis] = indices_list

        out_tensor = gathered_tensor.view(*(np.array(tensor.shape)[permuted_axis]))
        out_tensor = out_tensor.permute(*inverse_permutation)
        out_tensor = out_tensor.contiguous()

        # if axis != 2 and axis != 3:
        #     raise ValueError("Axis was "+str(axis))
        # if tensor.shape[dim] == 1:
        #     roll_0th_tensor = True
        #     shape = np.array(tensor.shape)
        #     shape[dim] = len(shift)
        #
        #     # # FOR TESTING
        #     # auxx_shape = [1] * len(tensor.shape)
        #     # auxx_shape[dim] = len(shift)
        #     # return tensor.repeat(*auxx_shape)
        #     # # TIL HERE
        #
        #     rolled_tensor = torch.empty(tuple(shape), device=tensor.device)
        # else:
        #     roll_0th_tensor = False
        #
        #     # # FOR TESTING
        #     # return tensor
        #     # # TIL HERE
        #
        #     rolled_tensor = torch.empty_like(tensor, device=tensor.device)
        #
        # if dim == 0:
        #
        #     for idx in range(len(shift)):
        #         if roll_0th_tensor:
        #             idx_in_tensor = 0
        #         else:
        #             idx_in_tensor = idx
        #
        #         # TEST. Should always be true! Unless we are shifting along D
        #         # assert tensor.shape[axis] == group_sz
        #         shifted_indices = (torch.arange(0, tensor.shape[axis], dtype=torch.long, device=tensor.device) - shift[idx].long()) % tensor.shape[axis]
        #         if axis == 2:
        #             rolled_tensor[idx] = tensor[[idx_in_tensor]][:, :, shifted_indices]
        #         elif axis == 3:
        #             rolled_tensor[idx] = tensor[[idx_in_tensor]][:, :, :, shifted_indices]
        #         # auxx = TransformTensorBatch._roll(tensor[[idx_in_tensor]], int(shift[idx]), axis)
        #         # assert torch.all(torch.eq(auxx, rolled_tensor[[idx]]))
        #     assert torch.all(torch.eq(out_tensor, rolled_tensor))
        # elif dim == 1:
        #
        #     for idx in range(len(shift)):
        #         if roll_0th_tensor:
        #             idx_in_tensor = 0
        #         else:
        #             idx_in_tensor = idx
        #
        #         # TEST. Should always be true! Unless we are shifting along D
        #         # assert tensor.shape[axis] == group_sz
        #         shifted_indices = (torch.arange(0, tensor.shape[axis], dtype=torch.long, device=tensor.device) - shift[idx].long()) % tensor.shape[axis]
        #         if axis == 2:
        #             rolled_tensor[:, [idx]] = tensor[:, [idx_in_tensor]][:, :, shifted_indices]
        #         elif axis == 3:
        #             rolled_tensor[:, [idx]] = tensor[:, [idx_in_tensor]][:, :, :, shifted_indices]
        #         # auxx = TransformTensorBatch._roll(tensor[[idx_in_tensor]], int(shift[idx]), axis)
        #         # assert torch.all(torch.eq(auxx, rolled_tensor[:, [idx]]))
        #     assert torch.all(torch.eq(out_tensor, rolled_tensor))
        # else:
        #     raise NotImplementedError
        return out_tensor

    @staticmethod
    def _roll(tensor, shift, axis):
        """
        rolls a tensor across axis, shift positions and returns a new tensor with the result.

        :param tensor: tensor, any tensor
        :param shift: int how many positions to roll
        :param axis: int across which axis to rooll
        :return: tensor with axis rolled shift positions
        """

        if shift == 0:
            return tensor

        if axis < 0:
            axis += tensor.dim()

        dim_size = tensor.size(axis)
        after_start = dim_size - shift
        if shift < 0:
            after_start = -shift
            shift = dim_size - abs(shift)

        before = tensor.narrow(axis, 0, dim_size - shift)
        after = tensor.narrow(axis, after_start, shift)
        return torch.cat([after, before], axis)

    def validate_input(self, rot_thetas, scale_thetas):
        if rot_thetas is not None:
            if type(rot_thetas) != torch.Tensor:
                raise ValueError("rot_thetas should be tensor")
            if rot_thetas.dtype != torch.float32:
                raise ValueError("rot_thetas should be of type float32")
            if max(rot_thetas) > 2 * np.pi or min(rot_thetas) < 0:
                raise ValueError("angles should be in range [0,2*pi)")
        if scale_thetas is not None:
            if type(scale_thetas) != torch.Tensor:
                raise ValueError("scale_thetas should be tensor")
            if scale_thetas.dtype != torch.float32:
                raise ValueError("scale_thetas should be of type float32")
            if max(scale_thetas) > 2 or min(scale_thetas) < 0:
                raise ValueError("scale angles should DEFINETLY be in range [0,2]")

    def rotate_and_scale_images(self, images, rot_thetas=None, scale_thetas=None):
        """
        rotates images according to angles

        :param images: tensor
            shape = [batch_sz, image_channels, image_w, image_h]
            a tensor representing a list of images
        :param rot_thetas: list(float)
            shape: batch_sz
            values should be in [0,2*pi)
        :param scale_thetas: list
            shape: batch_sz
            probably they should be between [0.5, 1.5]
        :param method: string
            'scipy' or 'gaussian'
        :return: a tensor representing a list of images, each rotated by the corresponding angle
            shape = [batch_sz, image_channels, image_w, image_h]
        """

        t_imgs = None
        if self.use_scipy_order2 is True:
            # this is old and bad
            rot_thetas = rot_thetas.detach().cpu().numpy()
            t_imgs = self._rotate_scipy(images=images, angles=rot_thetas, order=2)
        else:
            #self.validate_input(rot_thetas, scale_thetas)
            t_imgs = self._rotate_and_scale(images=images, rot_thetas=rot_thetas, scale_thetas=scale_thetas)
        return t_imgs

    @staticmethod
    def _rotate_scipy(images, angles, order):
        """
        rotates each image in images according to angles
        order is the polynomial order of the spline interpolation...probably

        :param images: tensor
            shape = [batch_sz, image_channels, image_w, image_h]
            a tensor representing a list of images
        :param angles: list of float
            an array of angles. For each image the rotation angle in degrees
        :param order: int
            0-5 order of interpolation
        :return: a tensor representing a list of images, each rotated by the corresponding angle
            shape = [batch_sz, image_channels, image_w, image_h]
        """
        if type(angles) != np.ndarray:
            angles = angles.numpy()
        angles = angles/(2*np.pi) * 360
        rot_imgs = torch.empty(images.shape, device=images.device)
        # set axis of rotation the last 2 axis
        dims = len(images.shape)
        axes = np.array([dims - 2, dims - 3])
        for idx in range(images.size()[0]):
            image = images.detach().cpu().numpy()[idx]
            rot_scipy_img = scipy.ndimage.interpolation.rotate(image, -angles[idx], axes=axes, order=order,
                                                               reshape=False)
            rot_img = torch.from_numpy(rot_scipy_img).to(device=rot_imgs.device)
            rot_imgs[idx] = rot_img
        return rot_imgs

    def _rotate(self, images, rot_thetas):
        """
        rotates each image in images according to angles
        order is the polynomial order of the spline interpolation...probably

        :param images: tensor
            shape = [batch_sz, image_channels, image_w, image_h]
            a tensor representing a list of images
        :param rot_thetas: list of float
            an array of angles. For each image the rotation angle in degrees
        :return: a tensor representing a list of images, each rotated by the corresponding angle
            shape = [batch_sz, image_channels, image_w, image_h]
        """
        grid = self._get_affine_grid(images.shape, rot_thetas=rot_thetas)
        rot_images = self.interpolator.grid_sample(images, grid)
        return rot_images

    def _scale(self, images, scale_thetas):
        raise NotImplementedError()

    def _rotate_and_scale(self, images, rot_thetas, scale_thetas):
        """
        rotates each image in images according to angles
        order is the polynomial order of the spline interpolation...probably

        :param images: tensor
            shape = [batch_sz, image_channels, image_w, image_h]
            a tensor representing a list of images
        :param rot_thetas: list of float
            an array of angles. For each image the rotation parameter
        :param rot_thetas: list of float
            an array of angles. For each image the scale parameter
        :return: a tensor representing a list of images, each rotated by the corresponding angle
            shape = [batch_sz, image_channels, image_w, image_h]
        """
        grid = self._get_affine_grid(images.shape, rot_thetas=rot_thetas, scale_thetas=scale_thetas)
        t_images = self.interpolator.grid_sample(images, grid)
        return t_images

    def _get_affine_grid(self, shape, rot_thetas=None, scale_thetas=None):
        """
        Apply affine transformation ( actually without translation, s.t. we keep the image center fixed )
         to a grid with shape :param shape:. Returns the position of the pixels after applying the transformation to them
        Given a batch of angles for rotation and scale theta, transforms each grid in the batch accordingly to its respective angles

        :param shape: list(int)
            [batch_sz, ch_in, image_h, image_w]
        :param rot_thetas: list(float)
            shape: [batch_sz]
            values should be in [0,2*pi)
        :param scale_thetas: list
            shape: [batch_sz]
            probably they should be between [0.5, 1.5]
        :return:
            shape = [batch_sz, 2, image_w, image_h]
        """
        if scale_thetas is None:
            # one rotation matrix for each image in batch
            thetas = torch.stack((torch.stack((torch.cos(rot_thetas),
                                               -torch.sin(rot_thetas)),
                                              dim=0),
                                  torch.stack((torch.sin(rot_thetas),
                                               torch.cos(rot_thetas)),
                                              dim=0)),
                                 dim=0).permute(2, 0, 1)
            grid = self._affine_grid(theta=thetas, size=shape)

        elif rot_thetas is None:
            # one scale matrix for each image in batch
            thetas = torch.stack((torch.stack((scale_thetas,
                                               torch.zeros_like(scale_thetas)),
                                              dim=0),
                                  torch.stack((torch.zeros_like(scale_thetas),
                                               scale_thetas),
                                              dim=0)),
                                 dim=0).permute(2, 0, 1)
            grid = self._affine_grid(theta=thetas, size=shape)

        else:
            # one rotation*scale matrix for each image in batch
            thetas = torch.stack((torch.stack((torch.cos(rot_thetas) * scale_thetas,
                                               -torch.sin(rot_thetas) * scale_thetas),
                                              dim=0),
                                  torch.stack((torch.sin(rot_thetas) * scale_thetas,
                                               torch.cos(rot_thetas) * scale_thetas),
                                              dim=0)),
                                 dim=0).permute(2, 0, 1)
            grid = self._affine_grid(theta=thetas, size=shape)

        return grid

    def _affine_grid(self, theta, size):
        """
        Creates a meshgrid of indices at pixel positions in an image with size :param size: after
        transforming by the affine tranformation matrix :param theta:
        We only allow for rotation and scale, therefore, each transformation matrix in :param theta:
        is 2x2
        :param theta: tensor
            shape = [batch_sz, 2, 2]
        :param size: torch.Size
            shape = [4], contains the shapes of the output and input tensor, [batch_sz, image_channels, image_w, image_h]
        :return: tensor
            shape = [batch_sz, 2, image_w, image_h]
            meshgrid of pixel location after applying theta
        """
        # put in shape (2, points)

        if size[-2] != self.image_size[-2] or size[-2] != self.image_size[-2]:
            IJ = self.IJ[:, :size[-2], :size[-1]]
            list_IJ = IJ.contiguous().view(2, -1)
            IJ_shape = IJ.shape
        else:
            list_IJ = self.IJ.view(2, -1)
            IJ_shape = self.IJ.shape

        # center image indices
        shift_i = (size[-2] - 1) / 2
        shift_j = (size[-1] - 1) / 2
        shift = torch.tensor([shift_i, shift_j], device=self.IJ.device)
        centered_list_IJ = list_IJ - shift.unsqueeze(-1)

        #apply transformation
        new_list_IJ = torch.matmul(theta, centered_list_IJ)

        # reverse centering
        new_list_IJ += shift.unsqueeze(-1)

        #add the batch shape and reshape to original shape
        new_IJ = new_list_IJ.view(theta.shape[0], *IJ_shape)
        return new_IJ
