import os

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from constants import DEVICE, LOG_INTERVAL
from util import Util


class BasisEquivConvLyer(torch.nn.Module):

    def __init__(self, basis_ae, transformer, in_shape, nr_basis, nr_filters, stride,
                 filter_sz, conv_padding, index, bias=True):
        """
        nr_group_elems should be touple if we want combined groups such as p4m
        also nr_group_elems shoud not contain the nr of translations

        in_shape = (K, S, in_sz, in_sz) = nr_filters_in, nr_group_elems_in,
        out_shape = (K', S', out_sz, out_sz) = nr_filters, nr_group_elems,
        """
        super(BasisEquivConvLyer, self).__init__()
        self.transformed_basis = None
        self.filter_coefs = None
        self.bias = None
        self.transformer = transformer
        self.index = index
        # if len(in_shape) != 4:
        #     raise ValueError("in_shape should be tuple (K x S x n x n)")
        self.in_shape = in_shape
        self.in_channels = self.in_shape[0]
        self.in_group_elements = self.in_shape[1]
        self.one_by_one_filters = None

        if nr_filters < 1:
            raise ValueError("number of filters cannot be < 1")
        self.nr_filters = nr_filters

        if stride < 1:
            raise ValueError("stride cannot be < 1")
        self.stride = stride

        self.filter_sz = filter_sz
        self.basis_ae = basis_ae if filter_sz != 1 else None

        self.nr_basis = nr_basis

        self.conv_padding = conv_padding
        nr_group_elems = 1 if self.transformer is None else self.transformer.group.nr_group_elems

        self.out_shape = (self.nr_filters, nr_group_elems)  # , int(self.in_shape[2] / stride), int(self.in_shape[3] / stride))
        self.initialize_layer(bias)
        self.reset_parameters()

    def initialize_layer(self, bias):
        """
        initialize parameters by empty tensor with specific shapes
        filter_coefs: shape K' x D x K x S
        bias: shape K'
        """
        if self.filter_sz == 1:
            self.one_by_one_filters = nn.Parameter(torch.empty(
                size=(self.nr_filters, self.in_channels, self.in_group_elements, 1, 1),
                device=DEVICE))
        else:
            self.filter_coefs = nn.Parameter(torch.empty(
                size=(self.nr_filters, self.nr_basis, self.in_channels, self.in_group_elements),
                device=DEVICE))

        if bias:
            self.bias = nn.Parameter(torch.empty(self.nr_filters, device=DEVICE))

    def reset_parameters(self):
        """
        resets parameters value to normal distribution from xavier initialization
        """
        if self.filter_sz != 1:
            basis_std = torch.tensor([1 / 10])  # self.basis_ae.basis._gamma if self.basis_ae.basis._gamma is not None else torch.tensor([1 / 10])
            var = 2 / (self.basis_ae.nr_basis * self.in_channels * basis_std * basis_std)
            std = np.sqrt(var)
            std = std[0] / 10  # without /10 it results in nan's when not using bnorm

            if self.filter_coefs is not None:
                self.filter_coefs.data.normal_(0, std)
        else:
            var = 2. / (self.nr_filters * self.in_channels)
            std = np.sqrt(var)
            self.one_by_one_filters.data.normal_(0, std)

        if self.bias is not None:
            self.bias.data.normal_(0, std)

    def group_convolve(self, filters, input):
        """
        convolves filters with input:
        filters are reshaped to: K'*S' x K*S x n x n
        inputs are reshaped to: batch_sz x K*S x n x n
        :param filters: tensor with filters shape K' x S' x K x S x n x n
        :param input: tensor with input:
            input shape : batch_sz x K x n x n
            #IF WE DON'T SUM OVER S then:
            input shape for layer > 1: batch_sz x K x S x n x n
        :return: tensor of activations of shape batch_sz x K' x S' x n x n
        """
        input_view = input.view(input.size()[0], self.in_shape[0] * self.in_shape[1], input.shape[-2], input.shape[-1])  # self.in_shape[2], self.in_shape[3])
        planar_filters = filters.view(self.transformer.group.nr_group_elems * self.nr_filters,
                                      self.in_shape[0] * self.in_shape[1],
                                      self.filter_sz, self.filter_sz)

        # apply planar convolution on the reshpaed input and filters
        y = F.conv2d(input_view, weight=planar_filters, bias=None, stride=self.stride,
                     padding=self.conv_padding, )

        batch_size, _, ny_out, nx_out = y.size()
        y = y.view(batch_size, self.nr_filters, self.transformer.group.nr_group_elems, ny_out, nx_out)
        return y

    def forward(self, input, t_input=None):
        # get all the filters
        filters = self.get_transformed_filters()

        y = self.group_convolve(filters, input)

        if self.bias is not None:
            bias = self.bias.view(self.nr_filters, 1, 1, 1)
            y = y + bias

        return y

    def get_transformed_filters(self):
        """
        compute the linear combination of the basis and the coefficients to get the filters
        coefs: K' x D x K x S
        basis: S' x D x n x n
        filters: K' x S' x K x S x n x n
        apply roll across S, 3rd dimension to get the true equivariant filters
        return: a new tensor containing all filters; shape (K' x S' x K x S x n x n)
        """
        if self.filter_sz != 1:
            self.basis_ae.normalize_basis()
            normalized_basis = self.basis_ae.get_normalized_basis()
            filters = (self.filter_coefs[:, None, :, :, :, None, None] *
                       normalized_basis[:, :, None, None, :, :]).sum(2)
        else:
            filters = self.one_by_one_filters[:, None, :, :, :, :].repeat(1, self.transformer.group.nr_group_elems, 1, 1, 1, 1)
        if not self.in_group_elements == 1:
            filters = self.transformer.apply_roll(filters, 3,
                                                  torch.arange(0, self.transformer.group.nr_group_elems,
                                                               dtype=torch.long, device=DEVICE),
                                                  dim=1)
            # I could create the same output by rotating the coefficients, which is actually rolling of the coefficients
            # on a new axis
            # I could also roll the coefficients as follows:
            # braodcast them along S'
            # coefs: K' x S' x D x K x S
            # roll coefs along S
            # multiply this with the basis
            # since D = n*x usually, because we areusing a complete basis, it is the same amount of computation

            # aux_coefs = self.filter_coefs[:, None, :, :].repeat(1, self.transformer.group.nr_group_elems, 1, 1, 1)
            # aux_coefs = self.transformer.apply_roll(aux_coefs, 4,
            #             torch.arange(0, self.transformer.group.nr_group_elems,
            #                          dtype=torch.long, device=DEVICE),
            #             dim=1)
            # filters_aux = (aux_coefs[:, :, :, :, :, None, None] *
            #                normalized_basis[:, :, None, None, :, :]).sum(2)
            # assert torch.all(torch.eq(filters_aux, filters))

        return filters

    # def train_basis(self, data, equiv_rate, orthg_rate, verbose, epoch, batch_idx, model_name, dataset_len,
    #                 train_loader_len):
    #     loss = TrainerBasisAE.train_layer_batch(basis_ae=self.basis_ae,
    #                                             data=data,
    #                                             equiv_rate=equiv_rate,
    #                                             orthg_rate=orthg_rate,
    #                                             verbose=verbose,
    #                                             epoch=epoch,
    #                                             batch_idx=batch_idx,
    #                                             model_name=model_name,
    #                                             dataset_len=dataset_len,
    #                                             train_loader_len=train_loader_len)
    #
    #     return loss

    def plot(self, activations, model_name, epoch, batch_idx):
        with torch.no_grad():
            path_to_layer_folder = os.path.join('images', model_name, 'layer:' + str(self.index))
            path_to_layer_images = os.path.join(path_to_layer_folder, 'images')

            fig_number = 'epoch:' + str(epoch) + '_batch:' + str(batch_idx) + '_' + str(self.index) + '_'

            Util.show_activations(activations=activations, fig_name=fig_number,
                                  path_to_layer_images=path_to_layer_images)
            if self.filter_sz != 1:
                Util.plot_filters(self, save=True, fig_name=fig_number, path_to_layer_images=path_to_layer_images)

    def freeze_basis(self):
        if self.filter_sz != 1:
            self.basis_ae.freeze_basis()

    def extra_repr(self):
        s = 'nr_filters={nr_filters}, nr_basis={nr_basis}, ' \
            'in_channels={in_channels}, stride=({stride}, {stride}), ' \
            'padding=({conv_padding},{conv_padding})'
        if self.filter_sz == 1:
            s += ', filter_sz=(1,1)'
        return s.format(**self.__dict__)