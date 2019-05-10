import math

import torch

from util import Loss
from constants import DEVICE
from group_action import Group, RotationGroupTransformer
from interpolator.kernels import GaussianKernel, BilinearKernel
from transform_tensor_batch import TransformTensorBatch
from util import Util
import numpy as np


class WeilerBasis(torch.nn.Module):
    '''
    is set to Weiler basis of 3x3
    '''

    def forward(self, *input):
        raise NotImplementedError

    def __init__(self, basis_sz, nr_basis, nr_group_elems):
        super(WeilerBasis, self).__init__()
        assert basis_sz == 3
        self.basis_sz = basis_sz
        self.nr_basis = nr_basis
        self.nr_group_elems = nr_group_elems
        self.FROZEN = True
        self._unnormalized_basis = None
        self.range_basis = torch.arange(0, self.nr_basis, device=DEVICE, dtype=torch.long)
        self.identities = torch.eye(self.nr_basis, device=DEVICE)
        self.reset_parameters()
        self._gamma = None
        self._beta = None
        self.normalize = None

    def reset_parameters(self):
        pi = torch.tensor(math.pi, dtype=torch.float)
        base_element = pi / (self.nr_group_elems / 2)
        theta = torch.tensor([[3, 2, 1], [4, 0, 0], [5, 6, 7]], dtype=torch.float)[:, :] * pi / 4
        g = (torch.tensor([float(i) for i in range(self.nr_group_elems)]) * base_element)[:, None, None, None]
        k = torch.tensor([float(i) for i in range(self.nr_basis)])[:, None, None]
        basis_set = k * (theta - g)
        basis_cos = torch.cos(basis_set)
        basis_cos[:, :, 1, 1] = 0.
        self._unnormalized_basis = torch.tensor(basis_cos, device=DEVICE)
        basis_sin = torch.sin(basis_set)
        basis_sin[:, :, 1, 1] = 0.
        self._unnormalized_basis[:, 5:8] = torch.tensor(basis_sin, device=DEVICE)[:, 5:8]
        self._unnormalized_basis = self._unnormalized_basis

    def normalize_basis(self):
        """
        # self._unnormalized_basis.shape = S, D, flt_sz, flt_sz
        """
        if self.FROZEN:
            self._normalized_basis = self._unnormalized_basis  # + 0
        else:
            raise NotImplementedError

    def get_normalized_basis(self):
        return self._normalized_basis

    def get_normalized_basis_at_transformation_indices(self, indices):
        return self._normalized_basis[indices]

    def get_orthogonal_basis_loss(self):
        return Loss.get_orthogonal_basis_loss(self._normalized_basis, self.range_basis, self.identities)

    def freeze_basis(self):
        if self.FROZEN:
            return False

    def plot(self, fig_name, path_to_layer_images):
        self.normalize_basis()
        self._plot_basis(fig_name, path_to_layer_images)
        self._plot_orthogonality(fig_name, path_to_layer_images)

    def _plot_orthogonality(self, fig_name, path_to_layer_folder):
        Util.plot_orthogonality(self._normalized_basis, fig_name=fig_name, path_to_layer_images=path_to_layer_folder)

    def _plot_basis(self, fig_name, path_to_layer_images):
        Util.plot_basis(self._normalized_basis, fig_name=fig_name, path_to_layer_images=path_to_layer_images,
                        gamma=self._gamma,
                        beta=self._beta)

    def extra_repr(self):
        s = 'nr_basis={}, basis_sz=({},{})'.format(self.nr_basis, self.basis_sz, self.basis_sz)
        return s


class RandomBasis(torch.nn.Module):
    '''
    is set to random basis of 3x3
    '''

    def forward(self, *input):
        raise NotImplementedError

    def __init__(self, nr_basis, nr_group_elems, basis_sz):
        super(RandomBasis, self).__init__()
        assert basis_sz == 3
        self.basis_sz = basis_sz
        self.nr_basis = nr_basis
        self.nr_group_elems = nr_group_elems
        self.FROZEN = True
        self._unnormalized_basis = None
        self.range_basis = torch.arange(0, self.nr_basis, device=DEVICE, dtype=torch.long)
        self.identities = torch.eye(self.nr_basis, device=DEVICE)
        self.reset_parameters()
        self._gamma = None
        self._beta = None
        self.normalize = None

    def reset_parameters(self):
        self._unnormalized_basis = torch.nn.Parameter(
            torch.empty(self.nr_group_elems, self.nr_basis, 3, 3, device=DEVICE).data.normal_(0, 1),
            requires_grad=False)
        # basis_cos = np.cos(basis_set)
        # basis_cos[:, :, 1, 1] = 0.
        # self.basis_cos = basis_cos

    def normalize_basis(self):
        """
        # self._unnormalized_basis.shape = S, D, flt_sz, flt_sz
        """
        if self.FROZEN:
            self._normalized_basis = self._unnormalized_basis  # + 0
        else:
            raise NotImplementedError

    def get_normalized_basis(self):
        return self._normalized_basis

    def get_normalized_basis_at_transformation_indices(self, indices):
        return self._normalized_basis[indices]

    def get_orthogonal_basis_loss(self):
        return Loss.get_orthogonal_basis_loss(self._normalized_basis, self.range_basis, self.identities)

    def freeze_basis(self):
        if self.FROZEN:
            return False

    def plot(self, fig_name, path_to_layer_images):
        self.normalize_basis()
        self._plot_basis(fig_name, path_to_layer_images)
        self._plot_orthogonality(fig_name, path_to_layer_images)

    def _plot_orthogonality(self, fig_name, path_to_layer_folder):
        Util.plot_orthogonality(self._normalized_basis, fig_name=fig_name, path_to_layer_images=path_to_layer_folder)

    def _plot_basis(self, fig_name, path_to_layer_images):
        Util.plot_basis(self._normalized_basis, fig_name=fig_name, path_to_layer_images=path_to_layer_images,
                        gamma=self._gamma,
                        beta=self._beta)

    def extra_repr(self):
        s = 'nr_basis={}, basis_sz=({},{})'.format(self.nr_basis, self.basis_sz, self.basis_sz)
        return s


class Basis(torch.nn.Module):

    def forward(self, *input):
        raise NotImplementedError

    def __init__(self, normalize, nr_basis, basis_sz, nr_group_elems):
        super(Basis, self).__init__()
        self.nr_basis = nr_basis
        self.nr_group_elems = nr_group_elems
        self.normalize = normalize
        self.basis_sz = basis_sz
        self._unnormalized_basis = torch.nn.Parameter(
            torch.empty(nr_group_elems, nr_basis, basis_sz, basis_sz, device=DEVICE),
            requires_grad=True)
        self.range_basis = torch.arange(0, self.nr_basis, device=DEVICE, dtype=torch.long)
        self.identities = torch.eye(self.nr_basis, device=DEVICE)

        self._gamma = None
        self._beta = None
        if self.normalize is None:
            self._gamma = None
            self._beta = None
        elif self.normalize[0] is None:
            if self.normalize[1] == 'Experiment 1':
                self._gamma = torch.nn.Parameter(torch.tensor([1. / 45.], requires_grad=True, device=DEVICE))
                # self._beta = torch.tensor([0])
            elif self.normalize[1] == 'Experiment 2':
                self._gamma = torch.nn.Parameter(torch.tensor([1. / 45.], requires_grad=True, device=DEVICE))
                self._beta = torch.nn.Parameter(torch.tensor([0.], requires_grad=True, device=DEVICE))
            elif self.normalize[1] == 'Experiment 3':
                self._gamma = torch.nn.Parameter(torch.tensor([1. / 45.], requires_grad=True, device=DEVICE))
                self._beta = torch.nn.Parameter(torch.tensor([0.], requires_grad=True, device=DEVICE))
            elif self.normalize[1] == 'Experiment 4':
                self._gamma = torch.nn.Parameter(
                    torch.tensor([1. / 45.] * self.nr_basis, requires_grad=True, device=DEVICE))
                # self._beta = torch.tensor([0])
            else:
                raise ValueError("invalid Experiment: " + str(self.normalize[1]))

        elif self.normalize is not None:
            self._gamma, self._beta = self.get_normalization_schedule()

        self._normalized_basis = None
        self.FROZEN = False

    def reset_parameters(self):
        std = self._gamma[0].item() if self._gamma is not None else 1 / 45
        self._unnormalized_basis.data.normal_(0, std)

    def get_normalization_schedule(self):
        gamma_value, gamma_per_basis, gamma_learn, beta_per_basis, beta_learn = self.normalize

        if type(gamma_value) == int or type(gamma_value) == float:
            gamma = 1. / gamma_value
        else:
            raise ValueError

        if gamma_per_basis:
            gamma = [gamma] * self.nr_basis
        else:
            gamma = [gamma]

        if gamma_learn:
            gamma = torch.nn.Parameter(torch.tensor(gamma, device=DEVICE),
                                       requires_grad=True)
        else:
            gamma = torch.nn.Parameter(torch.tensor(gamma, device=DEVICE),
                                       requires_grad=False)

        beta = 0.

        if beta_per_basis:
            beta = [beta] * self.nr_basis
        else:
            beta = [beta]

        if beta_learn:
            beta = torch.nn.Parameter(torch.tensor(beta, device=DEVICE),
                                      requires_grad=True)
        else:
            beta = torch.nn.Parameter(torch.tensor(beta, device=DEVICE),
                                      requires_grad=False)

        return gamma, beta

    def normalize_basis(self):
        """
        # self._unnormalized_basis.shape = S, D, flt_sz, flt_sz
        """
        if self.FROZEN:
            self._normalized_basis = self._unnormalized_basis + 0
        else:
            if self.normalize is None:
                self._normalized_basis = self._unnormalized_basis + 0
            elif self.normalize[0] is None:
                if self.normalize[1] == 'Experiment 1':
                    l2_norm = self._unnormalized_basis.pow(2).sum(-1, keepdim=True).sum(-1, keepdim=True).sqrt()
                    self._normalized_basis = self._unnormalized_basis / l2_norm * self._gamma
                elif self.normalize[1] == 'Experiment 2':
                    l2_norm = self._unnormalized_basis.pow(2).sum(-1, keepdim=True).sum(-1, keepdim=True).sqrt()
                    self._normalized_basis = self._unnormalized_basis / l2_norm * self._gamma + self._beta
                elif self.normalize[1] == 'Experiment 3':
                    mean = self._unnormalized_basis.view(*self._unnormalized_basis.shape[:2], -1, ).mean(-1)
                    l2_norm = self._unnormalized_basis.pow(2).sum(-1, keepdim=True).sum(-1, keepdim=True).sqrt()
                    self._normalized_basis = (self._unnormalized_basis - mean[:, :, None,
                                                                         None]) / l2_norm * self._gamma + self._beta
                elif self.normalize[1] == 'Experiment 4':
                    l2_norm = self._unnormalized_basis.pow(2).sum(-1, keepdim=True).sum(-1, keepdim=True).sqrt()
                    self._normalized_basis = self._unnormalized_basis / l2_norm * self._gamma[:, None, None]
                else:
                    raise ValueError("invalid Experiment: " + str(self.normalize[1]))
            else:
                mean = self._unnormalized_basis.view(*self._unnormalized_basis.shape[:2], -1, ).mean(-1)
                std = self._unnormalized_basis.view(*self._unnormalized_basis.shape[:2], -1, ).std(-1)
                eps = 1e-12
                self._normalized_basis = self._gamma[:, None, None] * \
                                         (self._unnormalized_basis - mean[:, :, None, None]) / \
                                         (std[:, :, None, None] + eps) + \
                                         self._beta[:, None, None]

    def get_normalized_basis_at_transformation_indices(self, indices):
        return self._normalized_basis[indices]

    def get_normalized_basis(self):
        return self._normalized_basis

    def freeze_basis(self):
        if self.FROZEN:
            return False
        self.normalize_basis()
        self.FROZEN = True
        self._unnormalized_basis = torch.nn.Parameter(self._normalized_basis,
                                                      requires_grad=False)  # self._unnormalized_basis.detach().clone()
        self._gamma = None
        self._beta = None
        self.normalize = None

        # self._normalized_basis.requires_grad = False
        # for name, parameter in self.named_parameters():
        #     print("FROZEN: " + name)
        #     parameter.requires_grad = False
        return True

    def get_gamma_beta(self):
        return self._gamma, self._beta

    def get_orthogonal_basis_loss(self):
        return Loss.get_orthogonal_basis_loss(self._normalized_basis, self.range_basis, self.identities)

    def plot(self, fig_name, path_to_layer_images):
        self.normalize_basis()
        self._plot_basis(fig_name, path_to_layer_images)
        self._plot_orthogonality(fig_name, path_to_layer_images)

    def _plot_basis(self, fig_name, path_to_layer_images):
        Util.plot_basis(self._normalized_basis, fig_name=fig_name, path_to_layer_images=path_to_layer_images,
                        gamma=self._gamma,
                        beta=self._beta)

    def _plot_orthogonality(self, fig_name, path_to_layer_folder):
        Util.plot_orthogonality(self._normalized_basis, fig_name=fig_name, path_to_layer_images=path_to_layer_folder)

    def extra_repr(self):
        if self._gamma is not None and self._beta is not None:
            s = 'nr_basis={}, basis_sz=({},{}), gamma={}, gamma.requires_grad={}, ' \
                'gamma.shape={} beta={}, beta.requires_grad={}, ' \
                'beta.shape={}'.format(self.nr_basis, self.basis_sz, self.basis_sz, self._gamma.detach().cpu().numpy(),
                                       self._gamma.requires_grad, np.array(self._gamma.shape),
                                       self._beta.detach().cpu().numpy(), self._beta.requires_grad,
                                       np.array(self._beta.shape))
        elif self._gamma is not None:
            s = 'nr_basis={}, basis_sz=({},{}), gamma={}, gamma.requires_grad={}, ' \
                'gamma.shape={} beta={}'.format(self.nr_basis, self.basis_sz, self.basis_sz,
                                                self._gamma.detach().cpu().numpy(),
                                                self._gamma.requires_grad, np.array(self._gamma.shape),
                                                str(None))
        elif self._beta is not None:
            s = 'nr_basis={}, basis_sz=({},{}), gamma={}, beta={}, beta.requires_grad={}, ' \
                'beta.shape={}'.format(self.nr_basis, self.basis_sz, self.basis_sz, str(None),
                                       self._beta.detach().cpu().numpy(), self._beta.requires_grad,
                                       np.array(self._beta.shape))
        else:
            s = 'nr_basis={}, basis_sz=({},{}), gamma={}, beta={}'.format(self.nr_basis, self.basis_sz, self.basis_sz,
                                                                          str(None), str(None))

        return s


class AverageBasis(torch.nn.Module):

    def forward(self, *input):
        raise NotImplementedError

    def __init__(self, normalize, nr_basis, basis_sz, nr_group_elems):
        super(AverageBasis, self).__init__()
        self.nr_basis = nr_basis
        self.nr_group_elems = nr_group_elems
        self.normalize = normalize
        self.basis_sz = basis_sz
        self._unnormalized_basis = torch.nn.Parameter(
            torch.empty(nr_group_elems, nr_basis, basis_sz, basis_sz, device=DEVICE),
            requires_grad=True)
        self.range_basis = torch.arange(0, self.nr_basis, device=DEVICE, dtype=torch.long)
        self.identities = torch.eye(self.nr_basis, device=DEVICE)

        self._gamma = None
        self._beta = None
        if self.normalize is None:
            self._gamma = None
            self._beta = None
        elif self.normalize[0] is None:
            if self.normalize[1] == 'Experiment 1':
                self._gamma = torch.nn.Parameter(torch.tensor([1. / 45.], requires_grad=True, device=DEVICE))
                # self._beta = torch.tensor([0])
            elif self.normalize[1] == 'Experiment 2':
                self._gamma = torch.nn.Parameter(torch.tensor([1. / 45.], requires_grad=True, device=DEVICE))
                self._beta = torch.nn.Parameter(torch.tensor([0.], requires_grad=True, device=DEVICE))
            elif self.normalize[1] == 'Experiment 3':
                self._gamma = torch.nn.Parameter(torch.tensor([1. / 45.], requires_grad=True, device=DEVICE))
                self._beta = torch.nn.Parameter(torch.tensor([0.], requires_grad=True, device=DEVICE))
            elif self.normalize[1] == 'Experiment 4':
                self._gamma = torch.nn.Parameter(
                    torch.tensor([1. / 45.] * self.nr_basis, requires_grad=True, device=DEVICE))
                # self._beta = torch.tensor([0])
            else:
                raise ValueError("invalid Experiment: " + str(self.normalize[1]))

        elif self.normalize is not None:
            self._gamma, self._beta = self.get_normalization_schedule()

        self._normalized_basis = None
        self.FROZEN = False

    def reset_parameters(self):
        std = self._gamma[0].item() if self._gamma is not None else 1 / 45
        self._unnormalized_basis.data.normal_(0, std)

    def get_normalization_schedule(self):
        gamma_value, gamma_per_basis, gamma_learn, beta_per_basis, beta_learn = self.normalize

        if type(gamma_value) == int or type(gamma_value) == float:
            gamma = 1. / gamma_value
        else:
            raise ValueError

        if gamma_per_basis:
            gamma = [gamma] * self.nr_basis
        else:
            gamma = [gamma]

        if gamma_learn:
            gamma = torch.nn.Parameter(torch.tensor(gamma, device=DEVICE),
                                       requires_grad=True)
        else:
            gamma = torch.nn.Parameter(torch.tensor(gamma, device=DEVICE),
                                       requires_grad=False)

        beta = 0.

        if beta_per_basis:
            beta = [beta] * self.nr_basis
        else:
            beta = [beta]

        if beta_learn:
            beta = torch.nn.Parameter(torch.tensor(beta, device=DEVICE),
                                      requires_grad=True)
        else:
            beta = torch.nn.Parameter(torch.tensor(beta, device=DEVICE),
                                      requires_grad=False)

        return gamma, beta

    def normalize_basis(self):
        """
        # self._unnormalized_basis.shape = S, D, flt_sz, flt_sz
        """
        if self.FROZEN:
            self._normalized_basis = self._unnormalized_basis + 0
        else:
            if self.normalize is None:
                self._normalized_basis = self._unnormalized_basis + 0
            elif self.normalize[0] is None:
                if self.normalize[1] == 'Experiment 1':
                    l2_norm = self._unnormalized_basis.pow(2).sum(-1, keepdim=True).sum(-1, keepdim=True).sqrt()
                    self._normalized_basis = self._unnormalized_basis / l2_norm * self._gamma
                elif self.normalize[1] == 'Experiment 2':
                    l2_norm = self._unnormalized_basis.pow(2).sum(-1, keepdim=True).sum(-1, keepdim=True).sqrt()
                    self._normalized_basis = self._unnormalized_basis / l2_norm * self._gamma + self._beta
                elif self.normalize[1] == 'Experiment 3':
                    mean = self._unnormalized_basis.view(*self._unnormalized_basis.shape[:2], -1, ).mean(-1)
                    l2_norm = self._unnormalized_basis.pow(2).sum(-1, keepdim=True).sum(-1, keepdim=True).sqrt()
                    self._normalized_basis = (self._unnormalized_basis - mean[:, :, None,
                                                                         None]) / l2_norm * self._gamma + self._beta
                elif self.normalize[1] == 'Experiment 4':
                    l2_norm = self._unnormalized_basis.pow(2).sum(-1, keepdim=True).sum(-1, keepdim=True).sqrt()
                    self._normalized_basis = self._unnormalized_basis / l2_norm * self._gamma[:, None, None]
                else:
                    raise ValueError("invalid Experiment: " + str(self.normalize[1]))
            else:
                mean = self._unnormalized_basis.view(*self._unnormalized_basis.shape[:2], -1, ).mean(-1)
                std = self._unnormalized_basis.view(*self._unnormalized_basis.shape[:2], -1, ).std(-1)
                eps = 1e-12
                self._normalized_basis = self._gamma[:, None, None] * \
                                         (self._unnormalized_basis - mean[:, :, None, None]) / \
                                         (std[:, :, None, None] + eps) + \
                                         self._beta[:, None, None]

            group_elems_by_4 = int(self.nr_group_elems / 4)

            # data = data.transpose(-2, -1).flip(-2)  # 270*
            self._normalized_basis[group_elems_by_4:2 * group_elems_by_4] = self._normalized_basis[
                                                                            group_elems_by_4:2 * group_elems_by_4].transpose(
                -2, -1).flip(-2)
            # data = data.flip([-2, -1])  # 180*
            self._normalized_basis[2 * group_elems_by_4:3 * group_elems_by_4] = self._normalized_basis[
                                                                                2 * group_elems_by_4:3 * group_elems_by_4].flip(
                [-2, -1])
            # data = data.transpose(-2, -1).flip(-1)  # 90*
            self._normalized_basis[3 * group_elems_by_4:4 * group_elems_by_4] = self._normalized_basis[
                                                                                3 * group_elems_by_4:4 * group_elems_by_4].transpose(
                -2, -1).flip(-1)

            for i in range(group_elems_by_4):
                self._normalized_basis[i::group_elems_by_4] = self._normalized_basis[i::group_elems_by_4].mean(0,
                                                                                                               keepdim=True)

            # data = data.transpose(-2, -1).flip(-1)  # 90*
            self._normalized_basis[group_elems_by_4:2 * group_elems_by_4] = self._normalized_basis[
                                                                            group_elems_by_4:2 * group_elems_by_4].transpose(
                -2, -1).flip(-1)
            # data = data.flip([-2, -1])  # 180*
            self._normalized_basis[2 * group_elems_by_4:3 * group_elems_by_4] = self._normalized_basis[
                                                                                2 * group_elems_by_4:3 * group_elems_by_4].flip(
                [-2, -1])
            # data = data.transpose(-2, -1).flip(-2)  # 270*
            self._normalized_basis[3 * group_elems_by_4:4 * group_elems_by_4] = self._normalized_basis[
                                                                                3 * group_elems_by_4:4 * group_elems_by_4].transpose(
                -2, -1).flip(-2)

    def get_normalized_basis_at_transformation_indices(self, indices):
        return self._normalized_basis[indices]

    def get_normalized_basis(self):
        return self._normalized_basis

    def freeze_basis(self):
        if self.FROZEN:
            return False
        self.normalize_basis()
        self.FROZEN = True
        self._unnormalized_basis = torch.nn.Parameter(self._normalized_basis,
                                                      requires_grad=False)  # self._unnormalized_basis.detach().clone()
        self._gamma = None
        self._beta = None
        self.normalize = None
        # self._normalized_basis.requires_grad = False
        # for name, parameter in self.named_parameters():
        #     print("FROZEN: " + name)
        #     parameter.requires_grad = False
        return True

    def get_gamma_beta(self):
        return self._gamma, self._beta

    def get_orthogonal_basis_loss(self):
        return Loss.get_orthogonal_basis_loss(self._normalized_basis, self.range_basis, self.identities)

    def plot(self, fig_name, path_to_layer_images):
        self.normalize_basis()
        self._plot_basis(fig_name, path_to_layer_images)
        self._plot_orthogonality(fig_name, path_to_layer_images)

    def _plot_basis(self, fig_name, path_to_layer_images):
        Util.plot_basis(self._normalized_basis, fig_name=fig_name, path_to_layer_images=path_to_layer_images,
                        gamma=self._gamma,
                        beta=self._beta)

    def _plot_orthogonality(self, fig_name, path_to_layer_folder):
        Util.plot_orthogonality(self._normalized_basis, fig_name=fig_name, path_to_layer_images=path_to_layer_folder)

    def extra_repr(self):
        if self._gamma is not None and self._beta is not None:
            s = 'nr_basis={}, basis_sz=({},{}), gamma={}, gamma.requires_grad={}, ' \
                'gamma.shape={} beta={}, beta.requires_grad={}, ' \
                'beta.shape={}'.format(self.nr_basis, self.basis_sz, self.basis_sz, self._gamma.detach().cpu().numpy(),
                                       self._gamma.requires_grad, np.array(self._gamma.shape),
                                       self._beta.detach().cpu().numpy(), self._beta.requires_grad,
                                       np.array(self._beta.shape))
        elif self._gamma is not None:
            s = 'nr_basis={}, basis_sz=({},{}), gamma={}, gamma.requires_grad={}, ' \
                'gamma.shape={} beta={}'.format(self.nr_basis, self.basis_sz, self.basis_sz,
                                                self._gamma.detach().cpu().numpy(),
                                                self._gamma.requires_grad, np.array(self._gamma.shape),
                                                str(None))
        elif self._beta is not None:
            s = 'nr_basis={}, basis_sz=({},{}), gamma={}, beta={}, beta.requires_grad={}, ' \
                'beta.shape={}'.format(self.nr_basis, self.basis_sz, self.basis_sz, str(None),
                                       self._beta.detach().cpu().numpy(), self._beta.requires_grad,
                                       np.array(self._beta.shape))
        else:
            s = 'nr_basis={}, basis_sz=({},{}), gamma={}, beta={}'.format(self.nr_basis, self.basis_sz, self.basis_sz,
                                                                          str(None), str(None))

        return s


class GaussianInterpolatedBasis(torch.nn.Module):

    def forward(self, *input):
        raise NotImplementedError

    def __init__(self, normalize, nr_basis, basis_sz, nr_group_elems):
        super(GaussianInterpolatedBasis, self).__init__()
        kernel = GaussianKernel(3, 0.5)
        rotation_group = Group(name='rotation', nr_group_elems=nr_group_elems,
                               base_element=2 * math.pi / nr_group_elems)
        batch_transformer = TransformTensorBatch(kernel=kernel,
                                                 image_size=torch.Size((basis_sz + 2, basis_sz + 2)),
                                                 device=DEVICE,
                                                 group_sz=nr_group_elems,
                                                 use_scipy_order2=False)
        self.transformer = RotationGroupTransformer(group=rotation_group, device=DEVICE,
                                                    rotation_batch_tansformer=batch_transformer)
        self.nr_basis = nr_basis
        self.nr_group_elems = nr_group_elems
        self.normalize = normalize
        self.basis_sz = basis_sz
        self._unnormalized_basis = torch.nn.Parameter(
            torch.empty(nr_group_elems, nr_basis, basis_sz, basis_sz, device=DEVICE),
            requires_grad=True)
        self.range_nr_group_elems = torch.arange(0, self.transformer.group.nr_group_elems, device=DEVICE,
                                                 dtype=torch.float)
        self.range_basis = torch.arange(0, self.nr_basis, device=DEVICE, dtype=torch.long)
        self.identities = torch.eye(self.nr_basis, device=DEVICE)

        self._gamma = None
        self._beta = None
        if self.normalize is None:
            self._gamma = None
            self._beta = None
        elif self.normalize[0] is None:
            if self.normalize[1] == 'Experiment 1':
                self._gamma = torch.nn.Parameter(torch.tensor([1. / 45.], requires_grad=True, device=DEVICE))
                # self._beta = torch.tensor([0])
            elif self.normalize[1] == 'Experiment 2':
                self._gamma = torch.nn.Parameter(torch.tensor([1. / 45.], requires_grad=True, device=DEVICE))
                self._beta = torch.nn.Parameter(torch.tensor([0.], requires_grad=True, device=DEVICE))
            elif self.normalize[1] == 'Experiment 3':
                self._gamma = torch.nn.Parameter(torch.tensor([1. / 45.], requires_grad=True, device=DEVICE))
                self._beta = torch.nn.Parameter(torch.tensor([0.], requires_grad=True, device=DEVICE))
            elif self.normalize[1] == 'Experiment 4':
                self._gamma = torch.nn.Parameter(
                    torch.tensor([1. / 45.] * self.nr_basis, requires_grad=True, device=DEVICE))
                # self._beta = torch.tensor([0])
            else:
                raise ValueError("invalid Experiment: " + str(self.normalize[1]))

        elif self.normalize is not None:
            self._gamma, self._beta = self.get_normalization_schedule()

        self._normalized_basis = None
        self.FROZEN = False

    def reset_parameters(self):
        std = self._gamma[0].item() if self._gamma is not None else 1 / 45
        self._unnormalized_basis.data.normal_(0, std)

    def get_normalization_schedule(self):
        gamma_value, gamma_per_basis, gamma_learn, beta_per_basis, beta_learn = self.normalize

        if type(gamma_value) == int or type(gamma_value) == float:
            gamma = 1. / gamma_value
        else:
            raise ValueError

        if gamma_per_basis:
            gamma = [gamma] * self.nr_basis
        else:
            gamma = [gamma]

        if gamma_learn:
            gamma = torch.nn.Parameter(torch.tensor(gamma, device=DEVICE),
                                       requires_grad=True)
        else:
            gamma = torch.nn.Parameter(torch.tensor(gamma, device=DEVICE),
                                       requires_grad=False)

        beta = 0.

        if beta_per_basis:
            beta = [beta] * self.nr_basis
        else:
            beta = [beta]

        if beta_learn:
            beta = torch.nn.Parameter(torch.tensor(beta, device=DEVICE),
                                      requires_grad=True)
        else:
            beta = torch.nn.Parameter(torch.tensor(beta, device=DEVICE),
                                      requires_grad=False)

        return gamma, beta

    def normalize_basis(self):
        """
        # self._unnormalized_basis.shape = 1, D, flt_sz, flt_sz
        """
        if self.FROZEN:
            self._normalized_basis = self._unnormalized_basis + 0
        else:
            if self.normalize is None:
                self._normalized_basis = self._unnormalized_basis + 0
            elif self.normalize[0] is None:
                if self.normalize[1] == 'Experiment 1':
                    l2_norm = self._unnormalized_basis.pow(2).sum(-1, keepdim=True).sum(-1, keepdim=True).sqrt()
                    self._normalized_basis = self._unnormalized_basis / l2_norm * self._gamma
                elif self.normalize[1] == 'Experiment 2':
                    l2_norm = self._unnormalized_basis.pow(2).sum(-1, keepdim=True).sum(-1, keepdim=True).sqrt()
                    self._normalized_basis = self._unnormalized_basis / l2_norm * self._gamma + self._beta
                elif self.normalize[1] == 'Experiment 3':
                    mean = self._unnormalized_basis.view(*self._unnormalized_basis.shape[:2], -1, ).mean(-1)
                    l2_norm = self._unnormalized_basis.pow(2).sum(-1, keepdim=True).sum(-1, keepdim=True).sqrt()
                    self._normalized_basis = (self._unnormalized_basis - mean[:, :, None,
                                                                         None]) / l2_norm * self._gamma + self._beta
                elif self.normalize[1] == 'Experiment 4':
                    l2_norm = self._unnormalized_basis.pow(2).sum(-1, keepdim=True).sum(-1, keepdim=True).sqrt()
                    self._normalized_basis = self._unnormalized_basis / l2_norm * self._gamma[:, None, None]
                else:
                    raise ValueError("invalid Experiment: " + str(self.normalize[1]))
            else:
                mean = self._unnormalized_basis.view(*self._unnormalized_basis.shape[:2], -1, ).mean(-1)
                std = self._unnormalized_basis.view(*self._unnormalized_basis.shape[:2], -1, ).std(-1)
                eps = 1e-12
                self._normalized_basis = self._gamma[:, None, None] * \
                                         (self._unnormalized_basis - mean[:, :, None, None]) / \
                                         (std[:, :, None, None] + eps) + \
                                         self._beta[:, None, None]

            self._normalized_basis = self._normalized_basis[[0]]
            self._normalized_basis = self._normalized_basis.repeat(self.nr_group_elems, 1, 1, 1)
            self._normalized_basis = self.transformer.apply_sample_action_to_input(self._normalized_basis, self.range_nr_group_elems)

    def get_normalized_basis_at_transformation_indices(self, indices):
        return self._normalized_basis[indices]

    def get_normalized_basis(self):
        return self._normalized_basis

    def freeze_basis(self):
        if self.FROZEN:
            return False
        self.normalize_basis()
        self.FROZEN = True
        self._unnormalized_basis = torch.nn.Parameter(self._normalized_basis,
                                                      requires_grad=False)  # self._unnormalized_basis.detach().clone()
        self._gamma = None
        self._beta = None
        self.normalize = None
        # self._normalized_basis.requires_grad = False
        # for name, parameter in self.named_parameters():
        #     print("FROZEN: " + name)
        #     parameter.requires_grad = False
        return True

    def get_gamma_beta(self):
        return self._gamma, self._beta

    def get_orthogonal_basis_loss(self):
        return Loss.get_orthogonal_basis_loss(self._normalized_basis, self.range_basis, self.identities)

    def plot(self, fig_name, path_to_layer_images):
        self.normalize_basis()
        self._plot_basis(fig_name, path_to_layer_images)
        self._plot_orthogonality(fig_name, path_to_layer_images)

    def _plot_basis(self, fig_name, path_to_layer_images):
        Util.plot_basis(self._normalized_basis, fig_name=fig_name, path_to_layer_images=path_to_layer_images,
                        gamma=self._gamma,
                        beta=self._beta)

    def _plot_orthogonality(self, fig_name, path_to_layer_folder):
        Util.plot_orthogonality(self._normalized_basis, fig_name=fig_name, path_to_layer_images=path_to_layer_folder)

    def extra_repr(self):
        if self._gamma is not None and self._beta is not None:
            s = 'nr_basis={}, basis_sz=({},{}), gamma={}, gamma.requires_grad={}, ' \
                'gamma.shape={} beta={}, beta.requires_grad={}, ' \
                'beta.shape={}'.format(self.nr_basis, self.basis_sz, self.basis_sz, self._gamma.detach().cpu().numpy(),
                                       self._gamma.requires_grad, np.array(self._gamma.shape),
                                       self._beta.detach().cpu().numpy(), self._beta.requires_grad,
                                       np.array(self._beta.shape))
        elif self._gamma is not None:
            s = 'nr_basis={}, basis_sz=({},{}), gamma={}, gamma.requires_grad={}, ' \
                'gamma.shape={} beta={}'.format(self.nr_basis, self.basis_sz, self.basis_sz,
                                                self._gamma.detach().cpu().numpy(),
                                                self._gamma.requires_grad, np.array(self._gamma.shape),
                                                str(None))
        elif self._beta is not None:
            s = 'nr_basis={}, basis_sz=({},{}), gamma={}, beta={}, beta.requires_grad={}, ' \
                'beta.shape={}'.format(self.nr_basis, self.basis_sz, self.basis_sz, str(None),
                                       self._beta.detach().cpu().numpy(), self._beta.requires_grad,
                                       np.array(self._beta.shape))
        else:
            s = 'nr_basis={}, basis_sz=({},{}), gamma={}, beta={}'.format(self.nr_basis, self.basis_sz, self.basis_sz,
                                                                          str(None), str(None))

        return s


class BilinearInterpolatedBasis(torch.nn.Module):

    def forward(self, *input):
        raise NotImplementedError

    def __init__(self, normalize, nr_basis, basis_sz, nr_group_elems):
        super(BilinearInterpolatedBasis, self).__init__()
        kernel = BilinearKernel()
        rotation_group = Group(name='rotation', nr_group_elems=nr_group_elems,
                               base_element=2 * math.pi / nr_group_elems)
        batch_transformer = TransformTensorBatch(kernel=kernel,
                                                 image_size=torch.Size((basis_sz + 2, basis_sz + 2)),
                                                 device=DEVICE,
                                                 group_sz=nr_group_elems,
                                                 use_scipy_order2=False)
        self.transformer = RotationGroupTransformer(group=rotation_group, device=DEVICE,
                                                    rotation_batch_tansformer=batch_transformer)

        self.nr_basis = nr_basis
        self.nr_group_elems = nr_group_elems
        self.normalize = normalize
        self.basis_sz = basis_sz
        self._unnormalized_basis = torch.nn.Parameter(
            torch.empty(nr_group_elems, nr_basis, basis_sz, basis_sz, device=DEVICE),
            requires_grad=True)
        self.range_nr_group_elems = torch.arange(0, self.transformer.group.nr_group_elems, device=DEVICE,
                                                 dtype=torch.float)
        self.range_basis = torch.arange(0, self.nr_basis, device=DEVICE, dtype=torch.long)
        self.identities = torch.eye(self.nr_basis, device=DEVICE)

        self._gamma = None
        self._beta = None
        if self.normalize is None:
            self._gamma = None
            self._beta = None
        elif self.normalize[0] is None:
            if self.normalize[1] == 'Experiment 1':
                self._gamma = torch.nn.Parameter(torch.tensor([1. / 45.], requires_grad=True, device=DEVICE))
                # self._beta = torch.tensor([0])
            elif self.normalize[1] == 'Experiment 2':
                self._gamma = torch.nn.Parameter(torch.tensor([1. / 45.], requires_grad=True, device=DEVICE))
                self._beta = torch.nn.Parameter(torch.tensor([0.], requires_grad=True, device=DEVICE))
            elif self.normalize[1] == 'Experiment 3':
                self._gamma = torch.nn.Parameter(torch.tensor([1. / 45.], requires_grad=True, device=DEVICE))
                self._beta = torch.nn.Parameter(torch.tensor([0.], requires_grad=True, device=DEVICE))
            elif self.normalize[1] == 'Experiment 4':
                self._gamma = torch.nn.Parameter(
                    torch.tensor([1. / 45.] * self.nr_basis, requires_grad=True, device=DEVICE))
                # self._beta = torch.tensor([0])
            else:
                raise ValueError("invalid Experiment: " + str(self.normalize[1]))

        elif self.normalize is not None:
            self._gamma, self._beta = self.get_normalization_schedule()

        self._normalized_basis = None
        self.FROZEN = False

    def reset_parameters(self):
        std = self._gamma[0].item() if self._gamma is not None else 1 / 45
        self._unnormalized_basis.data.normal_(0, std)

    def get_normalization_schedule(self):
        gamma_value, gamma_per_basis, gamma_learn, beta_per_basis, beta_learn = self.normalize

        if type(gamma_value) == int or type(gamma_value) == float:
            gamma = 1. / gamma_value
        else:
            raise ValueError

        if gamma_per_basis:
            gamma = [gamma] * self.nr_basis
        else:
            gamma = [gamma]

        if gamma_learn:
            gamma = torch.nn.Parameter(torch.tensor(gamma, device=DEVICE),
                                       requires_grad=True)
        else:
            gamma = torch.nn.Parameter(torch.tensor(gamma, device=DEVICE),
                                       requires_grad=False)

        beta = 0.

        if beta_per_basis:
            beta = [beta] * self.nr_basis
        else:
            beta = [beta]

        if beta_learn:
            beta = torch.nn.Parameter(torch.tensor(beta, device=DEVICE),
                                      requires_grad=True)
        else:
            beta = torch.nn.Parameter(torch.tensor(beta, device=DEVICE),
                                      requires_grad=False)

        return gamma, beta

    def normalize_basis(self):
        """
        # self._unnormalized_basis.shape = S, D, flt_sz, flt_sz
        """
        if self.FROZEN:
            self._normalized_basis = self._unnormalized_basis + 0
        else:
            if self.normalize is None:
                self._normalized_basis = self._unnormalized_basis + 0
            elif self.normalize[0] is None:
                if self.normalize[1] == 'Experiment 1':
                    l2_norm = self._unnormalized_basis.pow(2).sum(-1, keepdim=True).sum(-1, keepdim=True).sqrt()
                    self._normalized_basis = self._unnormalized_basis / l2_norm * self._gamma
                elif self.normalize[1] == 'Experiment 2':
                    l2_norm = self._unnormalized_basis.pow(2).sum(-1, keepdim=True).sum(-1, keepdim=True).sqrt()
                    self._normalized_basis = self._unnormalized_basis / l2_norm * self._gamma + self._beta
                elif self.normalize[1] == 'Experiment 3':
                    mean = self._unnormalized_basis.view(*self._unnormalized_basis.shape[:2], -1, ).mean(-1)
                    l2_norm = self._unnormalized_basis.pow(2).sum(-1, keepdim=True).sum(-1, keepdim=True).sqrt()
                    self._normalized_basis = (self._unnormalized_basis - mean[:, :, None,
                                                                         None]) / l2_norm * self._gamma + self._beta
                elif self.normalize[1] == 'Experiment 4':
                    l2_norm = self._unnormalized_basis.pow(2).sum(-1, keepdim=True).sum(-1, keepdim=True).sqrt()
                    self._normalized_basis = self._unnormalized_basis / l2_norm * self._gamma[:, None, None]
                else:
                    raise ValueError("invalid Experiment: " + str(self.normalize[1]))
            else:
                mean = self._unnormalized_basis.view(*self._unnormalized_basis.shape[:2], -1, ).mean(-1)
                std = self._unnormalized_basis.view(*self._unnormalized_basis.shape[:2], -1, ).std(-1)
                eps = 1e-12
                self._normalized_basis = self._gamma[:, None, None] * \
                                         (self._unnormalized_basis - mean[:, :, None, None]) / \
                                         (std[:, :, None, None] + eps) + \
                                         self._beta[:, None, None]

            self._normalized_basis = self._normalized_basis[[0]]
            self._normalized_basis = self._normalized_basis.repeat(self.transformer.group.nr_group_elems, 1, 1, 1)
            self._normalized_basis = self.transformer.apply_sample_action_to_input(self._normalized_basis, self.range_nr_group_elems)

    def get_normalized_basis_at_transformation_indices(self, indices):
        return self._normalized_basis[indices]

    def get_normalized_basis(self):
        return self._normalized_basis

    def freeze_basis(self):
        if self.FROZEN:
            return False
        self.normalize_basis()
        self.FROZEN = True
        self._unnormalized_basis = torch.nn.Parameter(self._normalized_basis,
                                                      requires_grad=False)  # self._unnormalized_basis.detach().clone()
        self._gamma = None
        self._beta = None
        self.normalize = None
        # self._normalized_basis.requires_grad = False
        # for name, parameter in self.named_parameters():
        #     print("FROZEN: " + name)
        #     parameter.requires_grad = False
        return True

    def get_gamma_beta(self):
        return self._gamma, self._beta

    def get_orthogonal_basis_loss(self):
        return Loss.get_orthogonal_basis_loss(self._normalized_basis, self.range_basis, self.identities)

    def plot(self, fig_name, path_to_layer_images):
        self.normalize_basis()
        self._plot_basis(fig_name, path_to_layer_images)
        self._plot_orthogonality(fig_name, path_to_layer_images)

    def _plot_basis(self, fig_name, path_to_layer_images):
        Util.plot_basis(self._normalized_basis, fig_name=fig_name, path_to_layer_images=path_to_layer_images,
                        gamma=self._gamma,
                        beta=self._beta)

    def _plot_orthogonality(self, fig_name, path_to_layer_folder):
        Util.plot_orthogonality(self._normalized_basis, fig_name=fig_name, path_to_layer_images=path_to_layer_folder)

    def extra_repr(self):
        if self._gamma is not None and self._beta is not None:
            s = 'nr_basis={}, basis_sz=({},{}), gamma={}, gamma.requires_grad={}, ' \
                'gamma.shape={} beta={}, beta.requires_grad={}, ' \
                'beta.shape={}'.format(self.nr_basis, self.basis_sz, self.basis_sz, self._gamma.detach().cpu().numpy(),
                                       self._gamma.requires_grad, np.array(self._gamma.shape),
                                       self._beta.detach().cpu().numpy(), self._beta.requires_grad,
                                       np.array(self._beta.shape))
        elif self._gamma is not None:
            s = 'nr_basis={}, basis_sz=({},{}), gamma={}, gamma.requires_grad={}, ' \
                'gamma.shape={} beta={}'.format(self.nr_basis, self.basis_sz, self.basis_sz,
                                                self._gamma.detach().cpu().numpy(),
                                                self._gamma.requires_grad, np.array(self._gamma.shape),
                                                str(None))
        elif self._beta is not None:
            s = 'nr_basis={}, basis_sz=({},{}), gamma={}, beta={}, beta.requires_grad={}, ' \
                'beta.shape={}'.format(self.nr_basis, self.basis_sz, self.basis_sz, str(None),
                                       self._beta.detach().cpu().numpy(), self._beta.requires_grad,
                                       np.array(self._beta.shape))
        else:
            s = 'nr_basis={}, basis_sz=({},{}), gamma={}, beta={}'.format(self.nr_basis, self.basis_sz, self.basis_sz,
                                                                          str(None), str(None))

        return s
