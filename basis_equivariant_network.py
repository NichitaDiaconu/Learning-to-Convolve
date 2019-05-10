import torch
from torch import nn
from basis_equivariant_layer import BasisEquivConvLyer

# TODO move transformer HERE AND ADD CALLS TO TRANSFORMER FROM HERE
from basis_ae import BasisAE, TrainerBasisAE
from util import Util


class BasisEquivariantNet(nn.Module):
    def __init__(self, transformer, basis_equiv_layers, fc_sizes, shape_input, sz_output,
                 bias, pool_sz_conv, normalize_basis, stride_conv, lr, normalized_l2, onebyoneconv,
                 basis_equiv_layers_type, pool_type, last_layer_type):
        super(BasisEquivariantNet, self).__init__()
        self.last_layer_type = last_layer_type
        self.basis_equiv_layers_type = basis_equiv_layers_type
        self.pool_type = pool_type
        self.normalize_basis = normalize_basis
        self.pool_sz_conv = pool_sz_conv
        self.basis_equiv_layers = basis_equiv_layers
        self.len_non1_basis_equiv_layers =len([layer for layer in basis_equiv_layers if layer[2] != 1])
        self.sz_output = sz_output
        self.layers = nn.ModuleList()
        self.pool_sz = pool_sz_conv
        self.stride_conv = stride_conv
        if len(shape_input) == 3:
            if type != 'conv':
                # image has 1 transformation S
                # we add tuples (K, S) and (n, n) => (K, S, n, n)
                shape_input = tuple([shape_input[0], 1])
        else:
            raise ValueError("are we not sending images?")
        self.shape_input = shape_input

        # Add layers
        for idx, (nr_basis, nr_filters, filter_sz) in enumerate(basis_equiv_layers):
            if pool_type == 'stride':
                stride = stride_conv[idx]
            else:
                stride = 1
            if basis_equiv_layers_type != 'conv':
                if filter_sz != 1:
                    basis_ae_layer = BasisAE(in_shape=shape_input,
                                             nr_basis=nr_basis,
                                             transformer=transformer,
                                             basis_sz=filter_sz,
                                             padding=int(filter_sz / 2),
                                             normalize=normalize_basis,
                                             lr=lr, index=idx, normalized_l2=normalized_l2,
                                             basis_type=basis_equiv_layers_type)
                else:
                    basis_ae_layer = None
                layer = BasisEquivConvLyer(basis_ae=basis_ae_layer,
                                           transformer=transformer,
                                           in_shape=shape_input,
                                           nr_basis=nr_basis,
                                           nr_filters=nr_filters,
                                           stride=stride,  # stride_conv[idx],
                                           filter_sz=filter_sz,
                                           conv_padding=int(filter_sz / 2),
                                           bias=bias, index=idx)
                self.layers.append(layer)
                shape_input = layer.out_shape
                if stride_conv[idx] == 2:
                    if self.pool_type == 'avg':
                        self.layers.append(nn.AvgPool2d((2, 2), 2))
                    elif self.pool_type == 'max':
                        self.layers.append(nn.MaxPool2d((2, 2), 2))

                self.layers.append(nn.BatchNorm3d(nr_filters))
                self.layers.append(nn.ReLU())
            else:
                assert nr_basis is None
                assert filter_sz == 3
                layer = nn.Conv2d(shape_input[0], nr_filters, filter_sz, stride=stride, padding=1, bias=bias)
                shape_input = (nr_filters,)
                self.layers.append(layer)
                if stride_conv[idx] == 2:
                    if self.pool_type == 'avg':
                        self.layers.append(nn.AvgPool2d((2, 2), 2))
                    elif self.pool_type == 'max':
                        self.layers.append(nn.MaxPool2d((2, 2), 2))
                self.layers.append(nn.BatchNorm2d(nr_filters))
                self.layers.append(nn.ReLU())

        if len(onebyoneconv) != 0:
            if self.basis_equiv_layers_type != 'conv':
                self.layers.append(nn.AdaptiveAvgPool3d((1, None, None)))
            for sz in onebyoneconv:
                self.layers.append(nn.Conv1d(in_channels=shape_input[0], out_channels=sz, kernel_size=1))
                # if basis_equiv_layers_type != 'conv':
                #     shape_input = (sz, shape_input[1])
                #     self.layers.append(nn.BatchNorm3d(sz))
                # else:
                shape_input = (sz, )
                self.layers.append(nn.BatchNorm2d(sz))

                self.layers.append(nn.ReLU())

        if last_layer_type == 'conv1x1':
            self.layers.append(nn.Conv1d(in_channels=shape_input[0], out_channels=sz_output, kernel_size=1))
        if last_layer_type == 'group1x1':
            layer = BasisEquivConvLyer(basis_ae=None,
                                       transformer=transformer,
                                       in_shape=shape_input,
                                       nr_basis=0,
                                       nr_filters=sz_output,
                                       stride=1,  # stride_conv[idx],
                                       filter_sz=1,
                                       conv_padding=0,
                                       bias=bias, index=len(basis_equiv_layers))
            self.layers.append(layer)

        if self.pool_type == 'avg' or self.pool_type == 'stride':
            if self.basis_equiv_layers_type != 'conv' and len(onebyoneconv) == 0:
                self.layers.append(nn.AdaptiveAvgPool3d((1, 1, 1)))
            else:
                self.layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        elif self.pool_type == 'max':
            if self.basis_equiv_layers_type != 'conv' and len(onebyoneconv) == 0:
                self.layers.append(nn.AdaptiveMaxPool3d((1, 1, 1)))
            else:
                self.layers.append(nn.AdaptiveMaxPool2d((1, 1)))

        if len(fc_sizes) != 0:
            for sz in fc_sizes:
                self.layers.append(nn.Linear(shape_input[0], sz))
                self.layers.append(nn.BatchNorm1d(sz))
                self.layers.append(nn.ReLU())
                shape_input = (sz,)
            # self.layers.append(nn.Linear(shape_input[0], sz_output))
        if last_layer_type == 'linear':
            self.layers.append(nn.Linear(shape_input[0], sz_output))

    def forward_prediction(self, input, epoch, batch_idx, trainer_basis, writer=None, dataset_len=None, finetune_basis=False, eq_loss=False):
        if self.basis_equiv_layers_type != 'conv':
            x = input[:, :, None, :, :]
        else:
            x = input

        total_ae_loss = torch.tensor([0.], device=input.device)
        for idx, layer in enumerate(self.layers):
            # if not self.training and batch_idx == 0:
            #     writer.add_histogram('(' + str(idx) + '): ' + layer._get_name() + '/stats/val/data_hist',
            #                         x,
            #                         global_step=dataset_len * (epoch + 1))
            #     for name, param in layer.named_parameters():
            #         writer.add_histogram('({:s}): {:s}/stats/val/{:s}_hist'.format(str(idx), layer._get_name(), name),
            #                             param,
            #                             global_step=dataset_len * (epoch + 1))
            #
            #     if type(layer) == BasisEquivConvLyer:
            #         writer.add_histogram(
            #            '({:s}): {:s}/stats/val/{:s}_hist'.format(str(idx), layer._get_name(), 'filters'),
            #            layer.get_transformed_filters(),
            #            global_step=dataset_len * (epoch + 1))

            if type(layer) == BasisEquivConvLyer:

                if not finetune_basis and layer.filter_sz != 1 and \
                        epoch < trainer_basis.train_basis_last_epoch and \
                        batch_idx % trainer_basis.train_basis_every_n_batches == 0:
                    ae_loss = trainer_basis.train_layer_batch(basis_ae=layer.basis_ae, data=x.detach(),
                                                              epoch=epoch, batch_idx=batch_idx)
                    total_ae_loss += ae_loss

                y = layer(x)
                x = y

                # if self.stride_conv[layer.basis_ae.index] == 2:
                #     pool_padding = 0
                #     pool_kernel = (2, 2)
                #     if pool_kernel == (2, 2):
                #         new_shape = (y.shape[0], y.shape[1], y.shape[2],
                #                      int(y.shape[3] / 2),
                #                      int(y.shape[4] / 2))
                #     elif pool_kernel == (3, 3):
                #         new_shape = (y.shape[0], y.shape[1], y.shape[2],
                #                      int((y.shape[3] - (1 - pool_padding)) / 2),
                #                      int((y.shape[4] - (1 - pool_padding)) / 2))
                #     y = nn.MaxPool2d(pool_kernel, 2, padding=pool_padding)(y.view(-1, y.shape[-2], y.shape[-1])).view(new_shape)
                #     # y = nn.AvgPool2d(pool_kernel, 2, padding=pool_padding)(y.view(-1, y.shape[-2], y.shape[-1])).view(new_shape)

                if not finetune_basis and not self.training and batch_idx == -1:
                    trainer_basis.plot(layer, x, epoch, batch_idx)
                if self.sz_output != layer.nr_filters and not finetune_basis and not self.training and eq_loss and batch_idx == 0:
                    trainer_basis.l2_normalized_total_equivariance(layer, x, save=True, epoch=epoch, batch_idx=batch_idx)
                if finetune_basis and layer.index == len(self.basis_equiv_layers)-1:
                    return y

            elif type(layer) == nn.Conv1d:
                x_shape = x.shape
                x_1d_shape = (x_shape[0], x_shape[1], -1)
                x = layer(x.view(x_1d_shape))
                new_x_shape = (x_shape[0], layer.out_channels, *x_shape[2:])
                x = x.view(new_x_shape)

            elif type(layer) == nn.BatchNorm3d or type(layer) == nn.BatchNorm2d or type(layer) == nn.BatchNorm1d:
                # TODO remove this when we use BNORM
                # raise ValueError("Not using bnorm")
                x = layer(x)

            elif type(layer) == nn.Linear:
                # fc layer
                # # Sum over group elements
                # if len(x.shape) == 5:
                #     x = x.view(x.shape[0], x.shape[1], -1).mean(-1)  # group, height, width
                x = layer(x)
            elif type(layer) == nn.MaxPool2d or type(layer) == nn.AvgPool2d:
                new_shape = (*x.shape[:-2], int(x.shape[-2] / 2), int(x.shape[-1] / 2))
                y = layer(x.view(-1, x.shape[-2], x.shape[-1]))
                x = y
                x = x.view(new_shape)
            elif type(layer) == nn.AdaptiveAvgPool3d or type(layer) == nn.AdaptiveMaxPool3d or \
                    type(layer) == nn.AdaptiveMaxPool2d or type(layer) == nn.AdaptiveAvgPool2d:
                x = layer(x)
                x = x.squeeze()
            elif type(layer) == nn.Dropout3d or type(layer) == nn.Dropout2d:
                x = layer(x)
            elif type(layer) == nn.ReLU:
                x = layer(x)
            elif type(layer) == nn.Conv2d:
                x = layer(x)
            else:
                print(type(layer))
                raise NotImplementedError

        return x, total_ae_loss

    def forward_reconstruction(self, input, epoch, batch_idx, trainer_basis):
        x = input[:, :, None, :, :]
        for idx, layer in enumerate(self.layers):
            if type(layer) == BasisEquivConvLyer:
                if layer.filter_sz != 1:
                    ae_loss = trainer_basis.train_layer_batch(basis_ae=layer.basis_ae, data=x.detach(),
                                                              epoch=epoch, batch_idx=batch_idx)

                y = layer(x)

                if not self.training and batch_idx == -1:
                    # If we are plotting equivariance activations and the filters
                    trainer_basis.plot(layer, y, epoch, batch_idx)

                if len(self.basis_equiv_layers) == layer.basis_ae.index + 1:
                    return ae_loss

                x = y

            elif type(layer) == nn.BatchNorm3d:
                with torch.no_grad():
                    # TODO remove this when we use BNORM
                    # raise ValueError("Not using bnorm")
                    x = layer(x)

            elif type(layer) == nn.MaxPool2d:
                original_shape = x.shape
                x = layer(x.view(-1, original_shape[-2], original_shape[-1]))
                x = x.view(original_shape[0], original_shape[1], original_shape[2], x.shape[-2], x.shape[-1])

        raise ValueError("did not find layer for training recontruction")

    def forward(self, *input):
        raise NotImplementedError("")

    def freeze_basis(self):
        for layer in self.layers:
            if type(layer) == BasisEquivConvLyer and layer.filter_sz != 1:
                layer.freeze_basis()
