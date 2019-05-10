"""Various interpolation schemes for PyTorch"""
import torch


class Kernel(object):
    def __init__(self, kernel_type, kernel_width, kernel_height):
        self.kh = kernel_height
        self.kw = kernel_width
        self.kernel_type = kernel_type

    def get_weight(self, positions):
        raise NotImplementedError


class GaussianKernel(Kernel):
    """A continuous gaussian kernel"""

    def __init__(self, width, sigma):
        super(GaussianKernel, self).__init__('pixel_centered', width, width)
        self.sigma = sigma

    def get_weight(self, positions):
        """
        Return interpolation kernel weight for the offsets in :param positions:
        :param positions: tensor with offsets from mean for on the X axis and Y axis
            shape = [batch_sz, 2, image_h, image_w, kernel_h, kernel_w]
        :return: weights: a tensor of weights for each position
        """
        sigsq = self.sigma * self.sigma
        num = torch.exp(-0.5 * torch.sum(positions * positions, 1) / sigsq)
        den = torch.sum(torch.sum(torch.abs(num), -1, keepdim=True), -2, keepdim=True)
        return num / den


class BilinearKernel(Kernel):
    def __init__(self):
        super(BilinearKernel, self).__init__('pixel_surround', 2, 2)

    def get_weight(self, positions):
        """
        Return interpolation kernel weight for the offsets in :param positions:
        :param positions: tensor with offsets from mean for on the X axis and Y axis
            shape = [batch_sz, 2, image_h, image_w, kernel_h, kernel_w]
        :return: weights: a tensor of weights for each position
        """
        wi = torch.min(torch.abs(positions[:, 0, ...]), torch.ones(1, device=positions.device))
        wj = torch.min(torch.abs(positions[:, 1, ...]), torch.ones(1, device=positions.device))
        return (1 - wi) * (1 - wj)
