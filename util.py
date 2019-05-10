import os

import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torchvision import transforms, datasets
import torch.nn.functional as F
import numpy as np
from torchvision.utils import make_grid

from constants import BATCH_SIZE, PLOT_IMAGES


def get_data(dataset='MNIST', train_augment_angle='0'):
    if train_augment_angle == 'all':
        transform_augment = [transforms.RandomAffine(degrees=360, translate=None)]
    elif train_augment_angle == '0':
        transform_augment = [transforms.RandomAffine(degrees=0, translate=None)]
    else:
        augment_angle = int(train_augment_angle)
        transform_augment = [transforms.RandomChoice(
            [transforms.RandomAffine(degrees=(augment_angle * i, augment_angle * i), translate=None) for i in
             range(int(360 / augment_angle))])]

    if dataset == 'MNIST':
        transform_augment = [*transform_augment]
        transform_normalize = [transforms.Normalize((0.1307,), (0.3081,))]
        transform_train = transforms.Compose([*transform_augment, transforms.ToTensor(), *transform_normalize])
        transform_val = transforms.Compose([transforms.ToTensor(), *transform_normalize])

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./../data', train=True, download=True, transform=transform_train),
            batch_size=BATCH_SIZE, shuffle=True)  # , num_workers=4)
        validation_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./../data', train=False, transform=transform_val),
            batch_size=BATCH_SIZE, shuffle=False)

    elif dataset == 'CIFAR10':
        transform_augment = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                             *transform_augment]
        transform_normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        transform_train = transforms.Compose([*transform_augment, transforms.ToTensor(), transform_normalize])
        transform_val = transforms.Compose([transforms.ToTensor(), transform_normalize])

        train_loader = torch.utils.data.DataLoader(
            datasets.cifar.CIFAR10('./../data', train=True, download=True, transform=transform_train),
            batch_size=BATCH_SIZE, shuffle=True)  # , num_workers=4)
        validation_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./../data', train=False, transform=transform_val),
            batch_size=BATCH_SIZE, shuffle=False)
    else:
        raise NotImplementedError
    return train_loader, validation_loader


class Util:

    @staticmethod
    def create_and_save_figures(model, path):
        Util.plot_basis(model, save=True, path=path)
        Util.plot_normalized_basis(model, save=True, path=path)
        Util.plot_filters(model, save=True, path=path)

    @staticmethod
    def plot_basis(basis, gamma, beta, basis_idxs=None, transformation_idxs=None, save=True, fig_name='',
                   path_to_layer_images=''):
        """
        plots the basis in the first layer, which is expected to be a BasisEquivConvLyer of a model
        """
        if not PLOT_IMAGES:
            return
        S, D, flt_sz, flt_sz = basis.shape
        if basis_idxs is None:
            basis_idxs = np.arange(0, D)
        if transformation_idxs is None:
            transformation_idxs = np.arange(S)

        # FASTER FEWER DETAILS PLOTTING VERSION
        S, D, flt_sz, flt_sz = basis.shape
        basis = basis[transformation_idxs][:, basis_idxs].permute(1, 0, 2, 3)
        basis = basis.contiguous().view(len(basis_idxs) * len(transformation_idxs), 1, flt_sz, flt_sz)
        grid = make_grid(basis, nrow=len(transformation_idxs), normalize=True)
        npimg = grid.clone().detach().cpu().numpy()
        plt.figure()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.tight_layout()

        # # TODO CHANGE!!! TAKES LOADS OF TIME
        # # if layer.index == 0:
        # plt.figure(figsize=((4+4)*S, (2+4)*D))
        # for i1, chosen_basis_idx in enumerate(basis_idxs):
        #     for i2, chosen_transformation_idx in enumerate(transformation_idxs):
        #         plt.subplot(len(basis_idxs), len(transformation_idxs), i2 + 1 + i1 * len(transformation_idxs))
        #         auxx_1 = basis[
        #             chosen_transformation_idx, chosen_basis_idx].detach().cpu().numpy()
        #         plt.imshow(auxx_1, cmap='gray')
        #         cbar = plt.colorbar(fraction=0.05)
        #         plt.axis('off')
        #         plt.setp(cbar.ax.yaxis.get_ticklabels(), fontsize=18)
        #         if i2 == int(len(transformation_idxs) / 2):
        #             if beta is None:
        #                 mean = str(None)
        #             else:
        #                 mean = beta[chosen_basis_idx].item() if beta.shape[0] > 1 else beta[0].item()
        #                 mean = "{:5.5f}".format(mean)
        #             if gamma is None:
        #                 std = str(None)
        #             else:
        #                 std = gamma[chosen_basis_idx].item() if gamma.shape[0] > 1 else gamma[0].item()
        #                 std = "{:5.5f}".format(std)
        #             plt.title("basis {:2d}; t_idx {:2d}; mean {:s}; std {:s}".format(
        #                 chosen_basis_idx, chosen_transformation_idx, mean, std), fontsize=24)
        #
        # # plt.tight_layout()

        if save:
            aux_path = os.path.join(path_to_layer_images, str(fig_name) + 'basis.pdf')
            plt.savefig(aux_path, bbox_inches='tight')
        plt.close()
        # # else:
        # #     for i1, chosen_basis_idx in enumerate(basis_idxs):
        # #         plt.figure(figsize=(4 * layer.transformer.group.nr_group_elems, 4*layer.transformer.group.nr_group_elems))
        # #         for i2, chosen_transformation_idx in enumerate(transformation_idxs):
        # #             for i3, chosen_transformation_idx_2 in enumerate(transformation_idxs):
        # #                 plt.subplot(len(transformation_idxs), len(transformation_idxs), i3 + 1 + i2 * len(transformation_idxs))
        # #                 auxx_1 = layer.basis[
        # #                     chosen_transformation_idx, chosen_basis_idx, chosen_transformation_idx_2].detach().cpu().numpy()
        # #                 plt.imshow(auxx_1, cmap='gray')
        # #                 plt.colorbar(fraction=0.05)
        # #
        # #                 if i3 == int(len(transformation_idxs) / 2) and i2 == int(len(transformation_idxs) / 2):
        # #                     mean = beta[chosen_basis_idx].item() if beta.shape[0] > 1 else beta[0].item()
        # #                     std = gamma[chosen_basis_idx].item() if gamma.shape[0] > 1 else gamma[0].item()
        # #                     plt.title("basis {:2d}; mean {:5.5f}; std {:5.5f}".format(
        # #                         chosen_basis_idx, mean, std), fontsize=24)
        # #
        # #         # plt.tight_layout()
        # #         if save:
        # #             aux_path = os.path.join(path, folder_name, 'images', str(fig_name) + 'basis_'+str(i1))
        # #             plt.savefig(aux_path)
        # #         plt.close()

    @staticmethod
    def plot_orthogonality(basis, fig_name='', path_to_layer_images='', save=True):
        if not PLOT_IMAGES:
            return
        S, D, flt_sz, flt_sz = basis.shape
        vector_shape = (S * D, flt_sz * flt_sz)

        # vectorize each basis
        vector_basis = basis.view(vector_shape)
        # vector_basis = self.basis.permute(1, 0, 2, 3).contiguous().view(vector_shape)

        basisT_basis = torch.mm(vector_basis, vector_basis.transpose(0, 1))
        plt.imshow(basisT_basis.detach().cpu().numpy(), cmap='gray')
        plt.colorbar()
        if save:
            aux_path = os.path.join(path_to_layer_images, fig_name + 'ALL_basis_T_basis.pdf')
            plt.savefig(aux_path, bbox_inches='tight')
        plt.close()

        S, D, flt_sz, flt_sz = basis.shape
        vector_shape = (D, flt_sz * flt_sz)

        # vectorize each basis
        vector_basis = basis[[0]].view(vector_shape)
        # vector_basis = self.basis[:, [0]].permute(1, 0, 2, 3).view(vector_shape)
        basisT_basis = torch.mm(vector_basis, vector_basis.transpose(0, 1))
        plt.imshow(basisT_basis.detach().cpu().numpy(), cmap='gray')
        plt.colorbar()
        if save:
            aux_path = os.path.join(path_to_layer_images, fig_name + 'Basis0_basis_T_basis.pdf')
            plt.savefig(aux_path, bbox_inches='tight')
        plt.close()
        pass

    @staticmethod
    def plot_normalized_basis(model, basis_idxs=None, transformation_idxs=None, save=False, path=None):
        """
        plots the basis in the first layer, which is expected to be a BasisEquivConvLyer of a model
        """
        # layer = model.layers[0]
        # if basis_idxs is None:
        #     basis_idxs = np.arange(0, layer.nr_basis)
        # if transformation_idxs is None:
        #     transformation_idxs = np.arange(layer.transformer.group.nr_group_elems)
        #
        # basis = layer.get_normalized_basis(layer.transformed_basis)
        # plt.figure(figsize=(20, 20))
        # for i1, chosen_basis_idx in enumerate(basis_idxs):
        #     for i2, chosen_transformation_idx in enumerate(transformation_idxs):
        #         plt.subplot(len(basis_idxs), len(transformation_idxs), i2 + 1 + i1 * len(transformation_idxs))
        #         auxx_1 = basis[
        #             chosen_transformation_idx, chosen_basis_idx, 0, 0].detach().cpu().numpy()
        #         plt.imshow(auxx_1, cmap='gray')
        #         plt.colorbar(fraction=0.05)
        #
        #         if i2 == int(len(transformation_idxs) / 2):
        #             plt.title("basis " + str(chosen_basis_idx) + "; t_idx " + str(chosen_transformation_idx))
        #         else:
        #             plt.title("t_idx " + str(chosen_transformation_idx))
        # plt.tight_layout()
        # if save:
        #     path = os.path.join(path, 'normalized_basis.png')
        #     plt.savefig(path)
        # plt.close()
        pass

    @staticmethod
    def show_reconstruction(images, fig_name, path_to_layer_images='', save=True, nrow=10):
        if not PLOT_IMAGES:
            return
        grid = make_grid(images[:, [0]], nrow=int(images.shape[0]/nrow), normalize=False)
        npimg = grid.clone().detach().cpu().numpy()
        plt.figure()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        if nrow == 10:
            # plt.title('original--input--target--reconstruction--loss--loss_normalized_to_image_scale')
            if save:
                aux_path = os.path.join(path_to_layer_images, fig_name + 'reconstruction.pdf')
                plt.savefig(aux_path, bbox_inches='tight')
            plt.close()
        elif nrow == 7:
            # plt.title('equivariance_loss')
            if save:
                aux_path = os.path.join(path_to_layer_images, fig_name + 'equivariance_loss.pdf')
                plt.savefig(aux_path, bbox_inches='tight')
            plt.close()

    @staticmethod
    def plot_filters(layer, filter_idxs=None, transformation_idxs=None, save=False,
                     fig_name='', path_to_layer_images=''):
        if not PLOT_IMAGES:
            return
        filters = layer.get_transformed_filters()

        if layer.in_group_elements == 1:
            K_prime, S_prime, K, S, flt_sz, flt_sz = filters.shape

            if filter_idxs is None:
                filter_idxs = np.arange(0, S_prime)
            if transformation_idxs is None:
                transformation_idxs = np.arange(S_prime)

            selected_filters = filters[filter_idxs][:, transformation_idxs]
            selected_filters = selected_filters.contiguous().view(len(filter_idxs) * len(transformation_idxs), K, flt_sz, flt_sz)
            grid = make_grid(selected_filters, nrow=len(transformation_idxs), normalize=True, padding=1)
            npimg = grid.clone().detach().cpu().numpy()
            plt.figure()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.tight_layout()

            # plt.figure(figsize=((4+4)*layer.transformer.group.nr_group_elems, (2+4)*layer.nr_basis))
            # for i1, chosen_filter_idx in enumerate(filter_idxs):
            #     for i2, chosen_transformation_idx in enumerate(transformation_idxs):
            #         plt.subplot(len(filter_idxs), len(transformation_idxs), i2 + 1 + i1 * len(transformation_idxs))
            #         auxx_1 = filters[chosen_filter_idx, chosen_transformation_idx, 0, 0].detach().cpu().numpy()
            #         plt.imshow(auxx_1, cmap='gray')
            #         cbar = plt.colorbar(fraction=0.05)
            #         plt.setp(cbar.ax.yaxis.get_ticklabels(), fontsize=18)
            #         plt.axis('off')
            #         if i2 == int(len(transformation_idxs) / 2):
            #             plt.title("filter " + str(chosen_filter_idx) + "; t_idx " + str(chosen_transformation_idx), fontsize=24)
            #         else:
            #             plt.title("t_idx " + str(chosen_transformation_idx), fontsize=24)
            # plt.tight_layout()
            if save:
                aux_path = os.path.join(path_to_layer_images, fig_name + 'filters.pdf')
                plt.savefig(aux_path, bbox_inches='tight')
            plt.close()

        else:
            if filter_idxs is None:
                filter_idxs = np.arange(0, min(layer.nr_filters, 5))
            if transformation_idxs is None:
                transformation_idxs = np.arange(layer.transformer.group.nr_group_elems)

            K_prime, S_prime, K, S, flt_sz, flt_sz = filters.shape
            selected_filters = filters[filter_idxs][:, transformation_idxs].contiguous()

            for i1, chosen_filter_idx in enumerate(filter_idxs):
                plt.figure(figsize=((2+4) * layer.transformer.group.nr_group_elems, (2+4) * layer.transformer.group.nr_group_elems))

                a_filter = selected_filters[chosen_filter_idx, :, 0, :].view(len(transformation_idxs), 1, S, flt_sz, flt_sz)
                a_filter = a_filter.permute(0, 2, 1, 3, 4).contiguous().view(len(transformation_idxs)*S, 1, flt_sz, flt_sz)
                grid = make_grid(a_filter, nrow=len(transformation_idxs), normalize=True, padding=1)
                npimg = grid.clone().detach().cpu().numpy()
                plt.figure()
                plt.imshow(np.transpose(npimg, (1, 2, 0)))
                plt.tight_layout()

                # for i2, chosen_transformation_idx in enumerate(transformation_idxs):
                #     for i3, chosen_transformation_idx_2 in enumerate(transformation_idxs):
                #
                #         plt.subplot(len(transformation_idxs), len(transformation_idxs), i3 + 1 + i2 * len(transformation_idxs))
                #         auxx_1 = filters[chosen_filter_idx, chosen_transformation_idx, 0, chosen_transformation_idx_2].detach().cpu().numpy()
                #         plt.imshow(auxx_1, cmap='gray')
                #         cbar = plt.colorbar(fraction=0.05)
                #         plt.setp(cbar.ax.yaxis.get_ticklabels(), fontsize=18)
                #         plt.axis('off')
                #
                #         if i2 == int(len(transformation_idxs) / 2) and i3 == int(len(transformation_idxs) / 2):
                #             plt.title("filter " + str(chosen_filter_idx) + "; t_idx " + str(chosen_transformation_idx),
                #                       fontsize=24)
                #         else:
                #             plt.title("t_idx " + str(chosen_transformation_idx), fontsize=24)
                # plt.tight_layout()

                if save:
                    aux_path = os.path.join(path_to_layer_images , fig_name + 'filters_'+str(i1) + '.pdf')
                    plt.savefig(aux_path, bbox_inches='tight')
                plt.close()
        pass

    @staticmethod
    def show_activations(activations, filter_idxs=None, save=True,
                         fig_name='', path_to_layer_images=''):
        if not PLOT_IMAGES:
            return
        B, K, S, H, W = activations.shape
        if filter_idxs is None:
            replace = False if activations.shape[1] >= 3 else True
            filter_idxs = np.random.choice(activations.shape[1], 3, replace=replace)
        for filter_idx in filter_idxs:
            grid = make_grid(activations[:S, filter_idx].contiguous().view(-1, 1, H, W), nrow=S, normalize=True)
            npimg = grid.detach().cpu().numpy()
            plt.figure()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            if save:
                aux_path = os.path.join(path_to_layer_images, fig_name + 'activations_' + str(filter_idx) + '.pdf')
                plt.savefig(aux_path, bbox_inches='tight')
            plt.close()


class Loss:

    @staticmethod
    def get_normalized_l1_loss_at_non_zero_indices(mat1, mat2, normalized_l1, group_name=None):
        mat2 = mat2.detach()

        width = mat1.shape[-1]
        one_sixth_width = int(width / 6)
        if one_sixth_width != 0:
            mat1 = mat1[...,
                   one_sixth_width:-one_sixth_width,
                   one_sixth_width:-one_sixth_width]
            mat2 = mat2[...,
                   one_sixth_width:-one_sixth_width,
                   one_sixth_width:-one_sixth_width]

        if group_name == 'scale':
            mat1 = mat1[mat2 != 0]
            mat2 = mat2[mat2 != 0]

        l1 = torch.abs(mat1 - mat2)

        l1_per_pixel = l1.mean()

        if not normalized_l1:
            mat1 = mat1.detach()
            mat2 = mat2.detach()
            l1 = l1.detach()

        if group_name == 'scale':
            l1_norm_per_image_mean = torch.tensor([-1])

        else:
            l1_norm_mat1 = torch.abs(mat1)
            l1_norm_mat2 = torch.abs(mat2)
            l1_sum = l1.view(l1.shape[0], -1).sum(-1)
            l1_norm_mat1_sum = torch.sqrt(l1_norm_mat1.view(l1_norm_mat1.shape[0], -1).sum(-1))
            l1_norm_mat2_sum = torch.sqrt(l1_norm_mat2.view(l1_norm_mat2.shape[0], -1).sum(-1))

            l1_norm_per_image = l1_sum / (l1_norm_mat1_sum * l1_norm_mat2_sum + 1e-10)

            # To compute mean:

            l1_norm_per_image_mean = l1_norm_per_image.mean()
        return l1_per_pixel, l1_norm_per_image_mean

    @staticmethod
    def get_normalized_l2_loss_at_non_zero_indices(mat1, mat2, normalized_l2, group_name=None):
        mat2 = mat2.detach()

        width = mat1.shape[-1]
        one_sixth_width = int(width / 4)
        if one_sixth_width != 0:
            mat1 = mat1[..., one_sixth_width:-one_sixth_width, one_sixth_width:-one_sixth_width]
            mat2 = mat2[..., one_sixth_width:-one_sixth_width, one_sixth_width:-one_sixth_width]

        if group_name == 'scale':
            mat1 = mat1[mat2 != 0]
            mat2 = mat2[mat2 != 0]

        l2 = torch.pow(mat1 - mat2, 2)

        l2_per_pixel = l2.mean()

        if not normalized_l2:
            mat1 = mat1.detach()
            mat2 = mat2.detach()
            l2 = l2.detach()

        if group_name == 'scale':
            l2_norm_per_image_mean = torch.tensor([-1])

        else:
            l2_norm_mat1 = torch.pow(mat1, 2).view(mat1.shape[0], -1).sum(-1).clamp(1e-7).sqrt()
            l2_norm_mat2 = torch.pow(mat2, 2).view(mat2.shape[0], -1).sum(-1).clamp(1e-7).sqrt()
            l2_sum = l2.view(l2.shape[0], -1).sum(-1)
            # l2_norm_mat1_sum = torch.sqrt(l2_norm_mat1.view(l2_norm_mat1.shape[0], -1).sum(-1))
            # l2_norm_mat2_sum = torch.sqrt(l2_norm_mat2.view(l2_norm_mat2.shape[0], -1).sum(-1))

            l2_norm_per_image = l2_sum / (l2_norm_mat1 * l2_norm_mat2 + 1e-10)

            # To compute mean:

            l2_norm_per_image_mean = l2_norm_per_image.mean()

            unsummed_normalized_per_pixel_loss = l2 / (l2_norm_mat1 * l2_norm_mat2 + 1e-10).view(l2_norm_mat2.shape[0],
                                                                                                 *[1 for _ in range(len(l2.shape) - 1)])

        return l2_per_pixel, l2_norm_per_image_mean, unsummed_normalized_per_pixel_loss

    @staticmethod
    def get_unsummed_l1_at_non_zero_indices(mat1, mat2):
        width = mat1.shape[-1]
        one_sixth_width = int(width / 6)
        if one_sixth_width != 0:
            result = torch.zeros_like(mat1)
            l1 = torch.abs(mat1[..., one_sixth_width:-one_sixth_width, one_sixth_width:-one_sixth_width] -
                           mat2[..., one_sixth_width:-one_sixth_width, one_sixth_width:-one_sixth_width])
            # l1[mat1 * mat2 == 0] = 0
            result[..., one_sixth_width:-one_sixth_width, one_sixth_width:-one_sixth_width] = l1
        else:
            l1 = torch.abs(mat1 - mat2)
            # l1[mat1 * mat2 == 0] = 0
            result = l1
        return result

    @staticmethod
    def get_orthogonal_basis_loss(basis, range_basis, identities):
        """
        computes basis orthogonality loss
        :return: basis orthogonality loss
        """
        loss = torch.tensor(0, dtype=torch.float, device=basis.device)

        # All basis rotation
        # get shape of all basis
        S, D, flt_sz, flt_sz = basis.shape
        vector_shape = (D, flt_sz * flt_sz)
        for i in range(S):
            # vectorize each basis
            vector_normalized_basis = basis[[i]].view(vector_shape)

            basisT_basis = torch.mm(vector_normalized_basis, vector_normalized_basis.transpose(0, 1))

            # Diagonal should be 1
            # elementwise_mean prior to v1 Pytorch
            loss += torch.nn.L1Loss(reduction="mean")(basisT_basis, identities)

            # Let diagonal be whathever
            # basisT_basis[range_basis, range_basis] = 1
            # loss += torch.nn.L1Loss(reduction="elementwise_mean")(basisT_basis, identities)

            # Diagonal be at least 0.1
            # aux_identities = identities * 0.1
            # aux_target = 0.1 - basisT_basis[range_basis, range_basis]
            # aux_identities[range_basis, range_basis] = aux_target
            # aux_identities[aux_target<=0] = 0
            # loss += torch.nn.L1Loss(reduction="elementwise_mean")(basisT_basis, aux_identities)

        loss /= S
        return loss
