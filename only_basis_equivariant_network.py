import torch
import numpy as np
from torch import nn
from basis_ae import BasisAE

from constants import BATCH_SIZE


# TODO: make pool size parameter

class OnlyBasisEquivariantNet(nn.Module):
    def __init__(self, transformer, shape_input, basis_equiv_layers, basis_sz, nr_transformations   , stride,
                 pool_sz, normalize, lr):
        super(OnlyBasisEquivariantNet, self).__init__()

        self.transformer = transformer
        self.basis_equiv_layers = basis_equiv_layers

        self.basis_sz = basis_sz
        self.shape_input = shape_input
        self.stride = stride
        self.nr_transformations = nr_transformations

        self.layer = BasisAE(in_shape=shape_input,
                                         nr_basis=basis_equiv_layers,
                                         transformer=transformer,
                                         stride=stride, basis_sz=basis_sz,
                                         padding=int(basis_sz / 2),
                                         pool_sz=pool_sz, normalize=normalize, lr=lr)


    def forward(self, rotated_input, rotated_output, input_transformation_index, output_transformation_index):
        reconstruction, reconstruction_loss, equivariance_loss, orthogonality_loss = self.layer(rotated_input,
                                                                                                rotated_output,
                                                                                                input_transformation_index,
                                                                                                output_transformation_index)
        return reconstruction, reconstruction_loss, equivariance_loss, orthogonality_loss
