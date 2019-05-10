# coding=utf-8
import os
import time

from torch.nn import functional as F
from torch.nn.modules import Module
from torch.nn.modules.utils import _single, _pair, _triple
from basis import *
from torchvision.utils import make_grid

from constants import BATCH_SIZE, LOG_INTERVAL, IN_CHANNELS_RECONSTRUCTION, DEVICE
from util import Util, Loss


class _ConvNd(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class Conv2dFunctional(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2dFunctional, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            _pair(0), groups, bias)

    def forward(self, input, weight, bias):
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)


class _ConvTransposeMixin(object):
    def _output_padding(self, input, output_size):
        if output_size is None:
            return self.output_padding

        output_size = list(output_size)
        k = input.dim() - 2
        if len(output_size) == k + 2:
            output_size = output_size[-2:]
        if len(output_size) != k:
            raise ValueError(
                "output_size must have {} or {} elements (got {})"
                    .format(k, k + 2, len(output_size)))

        def dim_size(d):
            return ((input.size(d + 2) - 1) * self.stride[d] -
                    2 * self.padding[d] + self.kernel_size[d])

        min_sizes = [dim_size(d) for d in range(k)]
        max_sizes = [min_sizes[d] + self.stride[d] - 1 for d in range(k)]
        for size, min_size, max_size in zip(output_size, min_sizes, max_sizes):
            if size < min_size or size > max_size:
                raise ValueError((
                    "requested an output size of {}, but valid sizes range "
                    "from {} to {} (for an input of {})").format(
                    output_size, min_sizes, max_sizes, input.size()[2:]))

        return tuple([output_size[d] - min_sizes[d] for d in range(k)])


class ConvTranspose2dFunctional(_ConvTransposeMixin, _ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, dilation=1, bias=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        super(ConvTranspose2dFunctional, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            output_padding, groups, bias)

    def forward(self, input, weight, bias, output_size=None):
        output_padding = self._output_padding(input, output_size)
        return F.conv_transpose2d(
            input, weight, bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)


class BasisAE(torch.nn.Module):
    def __init__(self, in_shape, nr_basis, transformer, basis_sz,
                 padding, normalize, lr, normalized_l2, batch_size=BATCH_SIZE, index=None, basis_type='learned'):
        super(BasisAE, self).__init__()

        self.normalized_l2 = normalized_l2
        self.index = index
        self.transformer = transformer
        self.batch_size = batch_size
        self.in_shape = in_shape
        self.padding = padding
        self.basis_sz = basis_sz
        self.nr_basis = nr_basis
        self.in_channels = in_shape[0] if IN_CHANNELS_RECONSTRUCTION is None else IN_CHANNELS_RECONSTRUCTION
        self.in_shape = (self.in_channels, *in_shape[1:])
        self.in_group_elements = self.in_shape[1]
        if self.in_shape[1] == 1:
            self.first_layer = True
        else:
            self.first_layer = False

        self.intermediate_shape = (self.in_channels, self.nr_basis)
        self.out_shape = in_shape
        self.conv = Conv2dFunctional(in_channels=self.batch_size * self.in_channels * self.in_group_elements,
                                     out_channels=self.batch_size * self.in_channels * self.in_group_elements * self.nr_basis,
                                     kernel_size=basis_sz, padding=padding,
                                     groups=self.batch_size * self.in_channels * self.in_group_elements)

        self.conv_transpose = ConvTranspose2dFunctional(
            in_channels=self.batch_size * self.in_channels * self.in_group_elements * nr_basis,
            out_channels=self.batch_size * self.in_channels * self.in_group_elements * self.nr_basis,
            kernel_size=basis_sz, padding=0,
            groups=self.batch_size * self.in_channels * self.in_group_elements * self.nr_basis,
            output_padding=0)
        self.basis, self.optimizer = init_basis_by_type(basis_type, basis_sz, nr_basis, transformer.group.nr_group_elems, normalize)
        self.start = time.time()

    def get_preprocessed_basis(self, transformation_indices):
        """

        :param transformation_indices: the indices of the transfoermations for each element in the batch
        :return: a tensor with the basis at the transformation indices corresponding to transformation_indices
                the basis is also added K, S dimensions, across which the basis is repeated
        """
        # repeat the basis at the same angle for each channel_in
        basis_view = self.basis.get_normalized_basis_at_transformation_indices(
            transformation_indices)
        # Add S channels
        basis_view = basis_view.unsqueeze(1).repeat(1, self.in_group_elements, 1, 1, 1)
        # Add K channels
        basis_view = basis_view.unsqueeze(1).repeat(1, self.in_channels, 1, 1, 1, 1)
        # then add to 1st channel
        basis_view = basis_view.view(-1, 1, self.basis_sz, self.basis_sz)
        return basis_view

    def encode(self, rotated_input, input_transformation_index):
        """
        input: shape B, K, S, 32, 32
        basis: shape S, D, 5, 5
        basis[input_transformation_index]; the basis at the specified angles
        basis: shape B, D, 5, 5
        span the basis across 2 new dimensions K, S because we treat each channel in the input as a separate image
        basis: shape B, K, S, D, 5, 5

        convolve each input and each basis to get
        activations: shape B, K, S, D, 32, 32
        """
        self.input_sptial_dims = (rotated_input.shape[-2], rotated_input.shape[-1])
        rotated_input_view = rotated_input.view(1,
                                                self.batch_size * self.in_channels * self.in_group_elements,
                                                rotated_input.shape[-2], rotated_input.shape[-1])
        # self.in_shape[-2], self.in_shape[-1])

        input_basis_view = self.get_preprocessed_basis(input_transformation_index.long())

        activation = self.conv.forward(rotated_input_view, input_basis_view, bias=None)

        activation_view = activation.view(self.batch_size, self.in_channels, self.in_group_elements, self.nr_basis,
                                          activation.shape[-2], activation.shape[-1])
        # self.intermediate_shape[-2], self.intermediate_shape[-1])

        # # TEST: (This only works when run on GPU... fck logic)
        # B, K, S, D = self.batch_size, self.in_channels, self.in_group_elements, self.nr_basis
        # for b in range(B):
        #     t_idx = input_transformation_index[b].long()
        #     for k in range(K):
        #         for s in range(S):
        #             for d in range(D):
        #                 one_activation = F.conv2d(rotated_input[b, k, s][None, None, :, :],
        #                                           self.basis.get_normalized_basis()[t_idx, d][None, None, :, :],
        #                                           padding=self.padding)
        #                 assert torch.all(torch.eq(one_activation, activation_view[b, k, s, d][None, None, :, :]))
        # print("passed encode")
        return activation_view

    def decode(self, rotated_activation, output_transformation_index):
        """
        input: shape B, K, S, D, 32, 32
        basis: shape S, D, 5, 5
        basis[output_transformation_index]; the basis at the specified angles
        basis: shape B, D, 5, 5
        span the basis across 2 new dimensions K, S because we treat each channel in the input as a separate image
        basis: shape B, K, S, D, 5, 5

        transpose convolve each input and each basis to get the reconstructions from each individual basis
        reconstruction: shape B, K, S, D, 32, 32
        sum across D, the basis dimension to get the reconstruction of the input
        reconstruction: shape B, K, S, 32, 32
        """

        rotated_activation_view = rotated_activation.view(1,
                                                          self.batch_size * self.in_channels *
                                                          self.in_group_elements * self.nr_basis,
                                                          rotated_activation.shape[-2],
                                                          rotated_activation.shape[-1])

        output_basis_view = self.get_preprocessed_basis(output_transformation_index.long())
        reconstruction = self.conv_transpose(rotated_activation_view, output_basis_view, bias=None)
        reconstruction = reconstruction.view(
            (self.batch_size, self.in_channels, self.in_group_elements, self.nr_basis, *reconstruction.shape[-2:]))

        # # TEST: (This only works when run on CPU... fck logic)
        # B, K, S, D = self.batch_size, self.in_channels, self.in_group_elements, self.nr_basis
        # for b in range(B):
        #     t_idx = output_transformation_index[b].long()
        #     for k in range(K):
        #         for s in range(S):
        #             for d in range(D):
        #                 one_reconstruction = F.conv_transpose2d(rotated_activation[b, k, s, d][None, None, :, :],
        #                                                         self.basis.get_normalized_basis()
        #                                                         [t_idx, d][None, None,:, :],
        #                                                         stride=1, padding=0)
        #                 assert torch.all(torch.eq(one_reconstruction, reconstruction[b, k, s, d][None, None, :, :]))
        # print("passed decode")

        # TODO understand why you need this fix and remove it
        if self.input_sptial_dims[-2] + self.padding > reconstruction.shape[-2]:
            padding = self.padding - (self.input_sptial_dims[-2] + self.padding - reconstruction.shape[-2])
        else:
            padding = self.padding
        reconstruction = reconstruction[..., padding:-padding, padding:-padding]

        reconstruction = reconstruction.sum(3)

        return reconstruction

    def forward(self, rotated_input, rotated_output, input_transformation_index, output_transformation_index):
        self.normalize_basis()

        input = rotated_input
        activation = self.encode(input, (
                2 * input_transformation_index - output_transformation_index) % self.transformer.group.nr_group_elems)

        difference = (output_transformation_index - input_transformation_index)

        # APPLY ACTION ON ACTIVATIONS
        rotated_input_activation = self.transformer.apply_sample_action_to_input(activation,
                                                                                 difference %
                                                                                 self.transformer.group.nr_group_elems)
        if not self.first_layer:
            rotated_input_activation = self.transformer.apply_roll(rotated_input_activation, 2, difference % \
                                                                   self.transformer.group.nr_group_elems, dim=0)
            # rotated_input_activation = self.transformer.apply_roll(rotated_input_activation, 3, difference% \
            #          self.nr_basis, dim=0)

        reconstruction = self.decode(rotated_input_activation, input_transformation_index)

        rotated_output_activation = self.encode(rotated_output, input_transformation_index)
        # TODO added this:
        rotated_output_activation = self.transformer.apply_identity_action_to_input(rotated_output_activation)

        return reconstruction, rotated_output, rotated_input_activation, rotated_output_activation

    def normalize_basis(self):
        self.basis.normalize_basis()

    def get_normalized_basis(self):
        return self.basis.get_normalized_basis()

    def freeze_basis(self):
        changed = self.basis.freeze_basis()
        if changed:
            print("FREEZING PARAMETERS IN LAYER " + str(self.index))
        self.optimizer = None


class TrainerBasisAE:

    def __init__(self, equiv_rate, orthg_rate, verbose, dataset_len, train_loader_len, model_name, log_writer,
                 train_basis_last_epoch=None, train_basis_every_n_batches=None):

        self.model_name = model_name
        self.train_loader_len = train_loader_len
        self.dataset_len = dataset_len
        self.verbose = verbose
        self.orthg_rate = orthg_rate
        self.equiv_rate = equiv_rate

        self.rec_loss_sum = None
        self.rec_loss_norm_sum = None
        self.equiv_loss_sum = None
        self.equiv_loss_norm_sum = None
        self.writer = log_writer
        self.train_basis_last_epoch = train_basis_last_epoch
        self.train_basis_every_n_batches = train_basis_every_n_batches

    def train_layer_batch(self, basis_ae, data, epoch, batch_idx):
        sample_size = (BATCH_SIZE,)

        if data.shape[1] != basis_ae.in_channels:
            indices = torch.tensor(np.random.randint(0, data.shape[1], [basis_ae.in_channels]), dtype=torch.long,
                                   device=data.device)
            data_subset = data[:, indices]  # .clone()
        else:
            data_subset = data  # .clone()

        if basis_ae.training:
            basis_ae.optimizer.zero_grad()
        # basis_ae.basis_optimizer.zero_grad()
        # basis_ae.normalizer_optimizer.zero_grad()
        # Construct input
        input_transformation_index = basis_ae.transformer.get_random_sample(sample_size)
        basis_ae.transformer.set_elements_sample(input_transformation_index)
        rot_input = basis_ae.transformer.apply_sample_action_to_input(data_subset)
        if not basis_ae.first_layer:
            rot_input = basis_ae.transformer.apply_roll(rot_input, 2, input_transformation_index, dim=0)

        # Construct ground truth
        output_transformation_index = basis_ae.transformer.get_random_sample(sample_size)
        if basis_ae.transformer.group.name == 'scale':
            output_transformation_index[output_transformation_index < input_transformation_index] = \
                input_transformation_index[output_transformation_index < input_transformation_index]
        basis_ae.transformer.set_elements_sample(output_transformation_index)
        rot_output = basis_ae.transformer.apply_sample_action_to_input(data_subset)
        if not basis_ae.first_layer:
            rot_output = basis_ae.transformer.apply_roll(rot_output, 2, output_transformation_index, dim=0)

        [reconstruction, rotated_output, rotated_input_activation, rotated_output_activation] = basis_ae.forward(
            rot_input,
            rot_output,
            input_transformation_index,
            output_transformation_index)

        [loss, reconstruction_loss, reconstruction_loss_norm, equivariance_loss, equivariance_loss_norm,
         orthogonality_loss, l2_loss] = self.compute_loss(basis_ae, reconstruction, rotated_output,
                                                          rotated_input_activation, rotated_output_activation)

        if basis_ae.training:
            # basis_ae.optimizer.zero_grad()
            loss.backward()
            basis_ae.optimizer.step()
            basis_ae.optimizer.zero_grad()

            # basis_ae.basis_optimizer.zero_grad()
            # loss.backward(retain_graph=True)
            # basis_ae.basis_optimizer.step()
            # basis_ae.basis_optimizer.zero_grad()

            # # Added special optimizer for beta and gamma:
            # basis_ae.normalizer_optimizer.zero_grad()
            # equivariance_loss.backward()  # reconstruction_loss
            # basis_ae.normalizer_optimizer.step()
            # basis_ae.normalizer_optimizer.zero_grad()
            # torch.cuda.empty_cache()

        self.log(basis_ae, batch_idx, epoch, data_subset, reconstruction_loss, reconstruction_loss_norm,
                 equivariance_loss, equivariance_loss_norm, orthogonality_loss, l2_loss, rot_input, rot_output,
                 reconstruction, rotated_input_activation, rotated_output_activation)
        return loss

    def reset_log_sums(self, nr_layers):
        self.rec_loss_sum = [0 for _ in range(nr_layers)]
        self.rec_loss_norm_sum = [0 for _ in range(nr_layers)]
        self.equiv_loss_sum = [0 for _ in range(nr_layers)]
        self.equiv_loss_norm_sum = [0 for _ in range(nr_layers)]

    def get_log_sums(self):
        return self.rec_loss_sum, self.rec_loss_norm_sum, self.equiv_loss_sum, \
               self.equiv_loss_norm_sum

    def log(self, basis_ae, batch_idx, epoch, data_subset, reconstruction_loss, reconstruction_loss_norm,
            equivariance_loss, equivariance_loss_norm, orthogonality_loss, l2_loss, rot_input, rot_output,
            reconstruction, rotated_input_activation, rotated_output_activation):
        if basis_ae.training:
            if ((batch_idx == 0 and epoch == 0) or (batch_idx + 1) % LOG_INTERVAL == 0):
                str_output = '[Train: Layer:{} Epoch:{} {}/{} ({:.0f}%)]:\tRec Loss per pixel:{:.6f}\t Norm Rec Loss per image:{'':.6f} \tEquiv Loss per pixel:{:.6f}\tNorm Equiv Loss per image:{:.6f}\tOrthg Loss:{:.6f}\tsum ''squared basis(l2 basis):{:.6f}'.format(
                    basis_ae.index,
                    epoch, batch_idx * len(data_subset), self.dataset_len,
                           100. * batch_idx / self.train_loader_len,
                    reconstruction_loss.item(),
                    reconstruction_loss_norm.item(),
                    equivariance_loss.item(),
                    equivariance_loss_norm.item(),
                    orthogonality_loss.item(),
                    l2_loss.item()
                )
                if self.verbose:
                    print(str_output)
                path_to_layer_folder = os.path.join('images', self.model_name, 'basis_layer:' + str(basis_ae.index))

                output_file = os.path.join(path_to_layer_folder, 'output.txt')

                with open(output_file, "a+") as f:
                    f.write(str_output)
                    f.write("\n")

            #     # TODO Log Here every N
            #     if basis_ae.basis._gamma is not None:
            #         if len(basis_ae.basis._gamma) == 1 and len(basis_ae.basis._gamma.shape) == 1:
            #             self.writer.add_scalar('basis_layer:' + str(basis_ae.index) + '/stats/train/gamma',
            #                                    basis_ae.basis._gamma,
            #                                    global_step=self.dataset_len * epoch + (batch_idx + 1) * BATCH_SIZE)
            #         else:
            #             self.writer.add_histogram('basis_layer:' + str(basis_ae.index) + '/stats/train/gamma',
            #                                      basis_ae.basis._gamma,
            #                                      global_step=self.dataset_len * epoch + (batch_idx + 1) * BATCH_SIZE)
            #
            #     if basis_ae.basis._beta is not None:
            #         if len(basis_ae.basis._beta) == 1 and len(basis_ae.basis._beta.shape) == 1:
            #             self.writer.add_scalar('basis_layer:' + str(basis_ae.index) + '/stats/train/beta',
            #                                    basis_ae.basis._beta,
            #                                    global_step=self.dataset_len * epoch + (batch_idx + 1) * BATCH_SIZE)
            #         else:
            #             self.writer.add_histogram('basis_layer:' + str(basis_ae.index) + '/stats/train/beta',
            #                                      basis_ae.basis._beta,
            #                                      global_step=self.dataset_len * epoch + (batch_idx + 1) * BATCH_SIZE)
            #
            # # TODO log here values at every step during training
            self.writer.add_scalar('basis_layer:' + str(basis_ae.index) + '/stats/train/reconstrution_loss',
                                   reconstruction_loss.item(),
                                   global_step=self.dataset_len * epoch + (batch_idx + 1) * BATCH_SIZE)
            self.writer.add_scalar('basis_layer:' + str(basis_ae.index) + '/stats/train/reconstruction_loss_norm',
                                   reconstruction_loss_norm.item(),
                                   global_step=self.dataset_len * epoch + (batch_idx + 1) * BATCH_SIZE)
            self.writer.add_scalar('basis_layer:' + str(basis_ae.index) + '/stats/train/equivariance_loss',
                                   equivariance_loss.item(),
                                   global_step=self.dataset_len * epoch + (batch_idx + 1) * BATCH_SIZE)
            self.writer.add_scalar('basis_layer:' + str(basis_ae.index) + '/stats/train/equivariance_loss_norm',
                                   equivariance_loss_norm.item(),
                                   global_step=self.dataset_len * epoch + (batch_idx + 1) * BATCH_SIZE)
            self.writer.add_scalar('basis_layer:' + str(basis_ae.index) + '/stats/train/orthogonality_loss',
                                   orthogonality_loss.item(),
                                   global_step=self.dataset_len * epoch + (batch_idx + 1) * BATCH_SIZE)
            self.writer.add_scalar('basis_layer:' + str(basis_ae.index) + '/stats/train/l2_loss',
                                   l2_loss.item(),
                                   global_step=self.dataset_len * epoch + (batch_idx + 1) * BATCH_SIZE)

        if not basis_ae.training:
            if batch_idx == 0:
                with torch.no_grad():
                    # On first batch plot
                    # TODO log every epoch statistics here
                    # self.writer.add_histogram('basis_layer:' + str(basis_ae.index) + '/stats/val/data_hist',
                    #                          data_subset,
                    #                          global_step=self.dataset_len * (epoch + 1))
                    # self.writer.add_histogram('basis_layer:' + str(basis_ae.index) + '/stats/val/rot_input_hist',
                    #                          rot_input,
                    #                          global_step=self.dataset_len * (epoch + 1))
                    # self.writer.add_histogram('basis_layer:' + str(basis_ae.index) + '/stats/val/rot_output_hist',
                    #                          rot_output,
                    #                          global_step=self.dataset_len * (epoch + 1))
                    # self.writer.add_histogram('basis_layer:' + str(basis_ae.index) + '/stats/val/reconstruction_hist',
                    #                          reconstruction,
                    #                          global_step=self.dataset_len * (epoch + 1))
                    # self.writer.add_histogram('basis_layer:' + str(basis_ae.index) + '/stats/val/normalized_basis_hist',
                    #                          basis_ae.basis.get_normalized_basis(),
                    #                          global_step=self.dataset_len * (epoch + 1))
                    # self.writer.add_histogram(
                    #    'basis_layer:' + str(basis_ae.index) + '/stats/val/rotated_input_activation_hist',
                    #    rotated_input_activation,
                    #    global_step=self.dataset_len * (epoch + 1))
                    # self.writer.add_histogram(
                    #    'basis_layer:' + str(basis_ae.index) + '/stats/val/rotated_output_activation_hist',
                    #    rotated_output_activation,
                    #    global_step=self.dataset_len * (epoch + 1))

                    self.print_validation(basis_ae, data_subset, rot_input, rot_output, reconstruction,
                                          epoch, batch_idx)
                    end = time.time()
                    # if self.verbose:
                    #     print('duration: ' + str(end - basis_ae.start))
                    basis_ae.start = end

            self.rec_loss_sum[basis_ae.index] += reconstruction_loss
            self.rec_loss_norm_sum[basis_ae.index] += reconstruction_loss_norm
            self.equiv_loss_sum[basis_ae.index] += equivariance_loss
            self.equiv_loss_norm_sum[basis_ae.index] += equivariance_loss_norm

    def compute_loss(self, basis_ae, reconstruction, rotated_output, rotated_input_activation,
                     rotated_output_activation):
        # Compute rec loss
        reconstruction_loss, reconstruction_loss_norm = Loss.get_normalized_l1_loss_at_non_zero_indices(reconstruction,
                                                                                                        rotated_output,
                                                                                                        normalized_l1=basis_ae.normalized_l2)
        # Compute equivariance loss
        equivariance_loss, equivariance_loss_norm = Loss.get_normalized_l1_loss_at_non_zero_indices(
            rotated_input_activation,
            rotated_output_activation, normalized_l1=basis_ae.normalized_l2)

        # Compute orthogonality loss
        orthogonality_loss = basis_ae.basis.get_orthogonal_basis_loss()

        loss = torch.tensor(0, dtype=torch.float, device=reconstruction_loss.device)

        if basis_ae.normalized_l2:
            loss += reconstruction_loss_norm
        else:
            loss += reconstruction_loss

        l2_loss = torch.pow(basis_ae.basis.get_normalized_basis(), 2).sum()
        if self.equiv_rate != 0 and self.equiv_rate is not None:
            if basis_ae.normalized_l2:
                loss += self.equiv_rate * equivariance_loss_norm
            else:
                loss += self.equiv_rate * equivariance_loss
        if self.orthg_rate != 0 and self.equiv_rate is not None:
            loss += self.orthg_rate * orthogonality_loss

        return loss, reconstruction_loss, reconstruction_loss_norm, equivariance_loss, \
               equivariance_loss_norm, orthogonality_loss, l2_loss

    def print_validation(self, basis_ae, data_subset, rot_input, rot_output, reconstruction, epoch, batch_idx):
        path_to_layer_folder = os.path.join('images', self.model_name, 'basis_layer:' + str(basis_ae.index))
        path_to_layer_images = os.path.join(path_to_layer_folder, 'images')

        samples = 10
        replace = False if data_subset.shape[0] >= samples else True
        sampled_indices = np.random.choice(data_subset.shape[0], samples, replace=replace)
        image_indices = torch.tensor(sampled_indices,
                                     device=rot_input.device,
                                     dtype=torch.long)

        fig_number = 'epoch:' + str(epoch) + '_batch:' + str(batch_idx) + '_' + str(basis_ae.index) + '_'

        loss_image = Loss.get_unsummed_l1_at_non_zero_indices(rot_output[image_indices],
                                                              reconstruction[image_indices])
        scaled_loss_image = loss_image.clone()
        im_min, im_max = (float(loss_image.min()), float(loss_image.max()))
        loss_image.add_(-im_min).div_(im_max - im_min + 1e-5)
        im_min = min(data_subset.min(), rot_input.min(), rot_output.min(), reconstruction.min())
        im_max = max(data_subset.max(), rot_input.max(), rot_output.max(), reconstruction.max())
        data_subset.add_(-im_min).div_(im_max - im_min + 1e-5)
        rot_input.add_(-im_min).div_(im_max - im_min + 1e-5)
        rot_output.add_(-im_min).div_(im_max - im_min + 1e-5)
        reconstruction.add_(-im_min).div_(im_max - im_min + 1e-5)
        scaled_loss_image.add_(-im_min).div_(im_max - im_min + 1e-5)

        if basis_ae.first_layer:
            images = torch.stack((data_subset[image_indices],
                                  rot_input[image_indices],
                                  rot_output[image_indices],
                                  reconstruction[image_indices],
                                  loss_image,
                                  scaled_loss_image), 1)
        else:
            images = torch.stack((data_subset[image_indices, :][:, :, [0]],
                                  rot_input[image_indices, :][:, :, [0]],
                                  rot_output[image_indices, :][:, :, [0]],
                                  reconstruction[image_indices, :][:, :, [0]],
                                  loss_image[:, :, [0]],
                                  scaled_loss_image[:, :, [0]]), 1)
        images = images.view(images.shape[0] * images.shape[1], images.shape[2], 1, images.shape[4],
                             images.shape[5]).squeeze(2)

        Util.show_reconstruction(images, fig_name=fig_number, path_to_layer_images=path_to_layer_images)

        basis_ae.basis.plot(fig_name=fig_number, path_to_layer_images=path_to_layer_images)

    def plot(self, layer, y, epoch, batch_idx):
        layer.plot(y, self.model_name, epoch, batch_idx)

    def reset_l2_normalized_total_equivariance_total_sums(self, nr_layers):
        self.l2_norm_total_equivariance = torch.tensor([0. for _ in range(nr_layers)], dtype=torch.float, device=DEVICE)

    def get_l2_normalized_total_equivariance_total_sums(self):
        return self.l2_norm_total_equivariance

    def l2_normalized_total_equivariance(self, layer, y, save=False, epoch=None, batch_idx=None):
        # we only care about the first 8 rotated images of the input, the others are just redundant
        backward_indices = torch.tensor([(-i % 8) for i in range(y.shape[0])], dtype=torch.float, device=y.device)
        # rotated_back_y = layer.transformer.apply_sample_action_to_input(y, backward_indices)
        if (layer.basis_ae is None and layer.filter_sz == 1) or type(layer.basis_ae.basis) in [Basis, AverageBasis,
                                                                                               BilinearInterpolatedBasis,
                                                                                               GaussianInterpolatedBasis,
                                                                                               RandomBasis]:
            rolled_rotated_back_y = layer.transformer.apply_roll(y, 2, backward_indices, 0)
        elif type(layer.basis_ae.basis) == WeilerBasis:
            rolled_rotated_back_y = layer.transformer.apply_roll(y, 2, (-backward_indices) % 8, 0)
        rolled_rotated_back_y = rolled_rotated_back_y[0:8]
        rolled_rotated_back_y = layer.transformer.apply_sample_action_to_input(rolled_rotated_back_y,
                                                                               backward_indices[0:8])
        new_y = rolled_rotated_back_y
        new_y = new_y[0:8]

        l2_per_pixel, l2_norm_per_image_mean, l2 = Loss.get_normalized_l2_loss_at_non_zero_indices(new_y[1:],
                                                                                                   new_y[[0]].repeat(
                                                                                                       new_y.shape[
                                                                                                           0] - 1,
                                                                                                       *[1] * (len(
                                                                                                           new_y.shape) - 1)),
                                                                                                   normalized_l2=True)
        # error = (new_y[[0]] - new_y[1:]).pow(2).view(7, -1).sum(-1)
        # error_norm = new_y[[0]].pow(2).view(1, -1).sum(-1).sqrt() * new_y[1:].pow(2).view(7, -1).sum(-1).sqrt()
        # normalized_equiv_at_layer = error / error_norm
        # l2_norm_per_image_mean = normalized_equiv_at_layer.sum() / 7

        self.l2_norm_total_equivariance[layer.index] += l2_norm_per_image_mean

        if save:
            im_min, im_max = l2.min(), l2.max()
            l2.add_(-im_min).div_(im_max - im_min + 1e-5)
            for filter_idx in range(5):
                fig_number = 'epoch:' + str(epoch) + '_batch:' + str(batch_idx) + '_' + str(
                    layer.index) + '_' + '_filter:' + str(filter_idx)
                path_to_layer_folder = os.path.join('images', self.model_name, 'layer:' + str(layer.index))
                path_to_layer_images = os.path.join(path_to_layer_folder, 'images')

                Util.show_reconstruction(
                    l2[:, filter_idx, :].contiguous().view(l2.shape[0] * l2.shape[2], 1, *(l2.shape[3:])),
                    fig_name=fig_number, path_to_layer_images=path_to_layer_images, nrow=7)

    """
    From plot activations:
        import matplotlib.pyplot as plt
        from torchvision.utils import make_grid
        
        new_y = rotated_subset
        # grid = make_grid((new_y[:8]-new_y[[0]]).abs().contiguous().view(-1, 1, new_y.shape[-2], new_y.shape[-1]), nrow=8, normalize=True)
        grid = make_grid(new_y[:8, 0].contiguous().view(-1, 1, new_y.shape[-2], new_y.shape[-1]), nrow=8, normalize=True)
        npimg = grid.detach().cpu().numpy()
        plt.figure()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    """


def init_basis_by_type(basis_type, basis_sz, nr_basis, nr_group_elems, normalize):
    if basis_type == 'learned':
        basis = Basis(normalize=normalize, nr_basis=nr_basis, basis_sz=basis_sz,
                      nr_group_elems=nr_group_elems)
        basis.reset_parameters()
        # lr was used before but it clashes with the other optimizer's lr and I found 0.003 to be fine
        optimizer = torch.optim.Adam(basis.parameters(), lr=0.003)
        # self.basis_optimizer = torch.optim.Adam([self.basis._unnormalized_basis, self.basis._beta], lr=lr)
        # self.normalizer_optimizer = torch.optim.Adam([self.basis._gamma], lr=lr)
    elif basis_type == 'weiler':
        basis = WeilerBasis(nr_basis=nr_basis, nr_group_elems=nr_group_elems,
                            basis_sz=basis_sz)
        basis.reset_parameters()
        optimizer = None
    elif basis_type == 'random':
        basis = RandomBasis(nr_basis=nr_basis, nr_group_elems=nr_group_elems,
                            basis_sz=basis_sz)
        optimizer = None
    elif basis_type == 'average':
        basis = AverageBasis(normalize=normalize, nr_basis=nr_basis, basis_sz=basis_sz,
                             nr_group_elems=nr_group_elems)
        basis.reset_parameters()
        optimizer = torch.optim.Adam(basis.parameters(), lr=0.003)
    elif basis_type == 'gaussian':
        basis = GaussianInterpolatedBasis(normalize=normalize, nr_basis=nr_basis, basis_sz=basis_sz,
                                          nr_group_elems=nr_group_elems)
        basis.reset_parameters()
        optimizer = None
    elif basis_type == 'bilinear':
        basis = BilinearInterpolatedBasis(normalize=normalize, nr_basis=nr_basis, basis_sz=basis_sz,
                                          nr_group_elems=nr_group_elems)
        basis.reset_parameters()
        optimizer = None
    else:
        raise NotImplementedError
    return basis, optimizer
