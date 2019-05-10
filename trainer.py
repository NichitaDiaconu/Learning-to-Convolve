import gc
import os
import shutil
import time
from types import SimpleNamespace
import csv
import numpy as np
import gc

import torch
import math

from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR

from basis_ae import TrainerBasisAE, Loss
from basis_equivariant_layer import BasisEquivConvLyer
from basis_equivariant_network import BasisEquivariantNet
from constants import BATCH_SIZE, TEST, LOG_INTERVAL, DEVICE, OVERFIT_SUBSET
from group_action import Group, RotationGroupTransformer, ScaleGroupTransformer
from interpolator.kernels import BilinearKernel, GaussianKernel
from serializer import Serializer
from transform_tensor_batch import TransformTensorBatch
from util import get_data


#
# def get_modelname(config_dict, config_dict):
#     modelname = ''
#     modelname += 'group_name:' + str(config_dict.group_name) + ' '
#     modelname += 'sigma:' + str(config_dict.sigma) + ' '
#     modelname += 'basis_sz:' + str(config_dict.basis_equiv_layers) + ' '
#     modelname += 'stride_sz_conv:' + str(config_dict.stride_sz_conv) + ' '
#     modelname += 'equivariance_rate:' + str(config_dict.equivariance_rate) + ' '
#     modelname += 'orthogonality_rate:' + str(config_dict.orthogonality_rate) + ' '
#     modelname += 'normalize:' + str(config_dict.normalize) + ' '
#     modelname += 'normalized_l2:' + str(config_dict.normalized_l2) + ' '
#     modelname += 'l2_coef:' + str(config_dict.weight_decay)
#     return modelname


def get_modelname(config_dict, target='prediction'):
    modelname = ''
    # I know, it puts _ after reconstruction and prediction; it is ok
    modelname += target + '_'
    modelname += 'dataset:' + str(config_dict.dataset) + '_'
    modelname += 'conv_type:' + str(config_dict.basis_equiv_layers_type) + '_'
    modelname += 'pool_type:' + str(config_dict.pool_type) + ' '
    modelname += 'last_layer_type:' + str(config_dict.last_layer_type) + ' '
    modelname += 'augmentation_angle:' + str(config_dict.rotation_augmentation_angle) + ' '
    modelname += 'aug_type :' + str(config_dict.rotation_augmentation_type) + ' '
    # modelname += 'sigma:' + str(config_dict.sigma) + ' '
    # modelname += 'basis_sz:' + str(config_dict.basis_equiv_layers) + ' '
    # modelname += 'stride_sz_conv:' + str(config_dict.stride_sz_conv) + ' '
    # modelname += 'equivariance_rate:' + str(config_dict.equivariance_rate) + ' '
    # modelname += 'orthogonality_rate:' + str(config_dict.orthogonality_rate) + ' '
    # modelname += 'normalize:' + str(config_dict.normalize) + ' '
    modelname += 'load:' + str(config_dict.load)

    modelname = modelname.replace(', ', ',')
    modelname = modelname.replace('Experiment ', 'Experiment', 1)

    return modelname


def init_trainer_model(equiv_rate, orthg_rate, lr, epochs, nr_group_elems, kernel_type,
                       width, sigma, basis_equiv_layers, fc_sizes,
                       bias, stride_sz_conv, normalize, weight_decay,
                       use_scipy_order2, group_name, save, save_aux, verbose,
                       train_basis_last_epoch, train_basis_every_n_batches, normalized_l2,
                       dataset, target, load, load_aux, onebyoneconv,
                       basis_equiv_layers_type, pool_type, last_layer_type, finetune_batches,
                       pool_sz_conv=None, sz_output=10, rotation_augmentation_angle=0, rotation_augmentation_type='torch', optimizer='adam'):
    """
    equiv_rate=1, orthg_rate=1, lr=0.003, epochs=10, nr_group_elems=4, kernel_type='Gaussian',
                       width=3, sigma=1., basis_equiv_layers=[(5, 20, 3)], fc_sizes=[2048], pool_sz_conv=None,
                       sz_output=10, bias=False, stride_sz_conv=1, normalize=False,
                       weight_decay=0, use_scipy_order2=False, group_name='rotation', save=False, save_aux=None,
                       verbose=True, train_basis_last_epoch=10000, train_basis_every_n_batches=1, normalized_l2=False,
                       dataset='CIFAR10', target='reconstruction', load=None, load_aux=None, onebyoneconv=[],
                       basis_equiv_layers_type='conv', pool_type='stride', last_layer_type='conv1x1', finetune_batches=1
    """
    assert rotation_augmentation_type in ['gaussian', 'torch']
    assert optimizer in ['adam', 'adam_noams', 'sgd']
    assert last_layer_type in ['conv1x1', 'group1x1', 'linear']
    assert pool_type in ['stride', 'avg', 'max']
    assert basis_equiv_layers_type in ['conv', 'random', 'weiler', 'learned', 'average', 'gaussian', 'bilinear']
    if pool_sz_conv is None:
        pool_sz_conv = [1 for _ in range(len(stride_sz_conv))]
    if kernel_type == 'Bilinear':
        kernel = BilinearKernel()
    elif kernel_type == 'Gaussian':
        kernel = GaussianKernel(width, sigma)
    else:
        raise ValueError('invalid parameter value')

    if target not in ['reconstruction', 'prediction']:
        raise ValueError('invalid parameter value target')

    if save:
        for el in save_aux:
            assert el in ['model', 'basis']

    if target == 'reconstruction':
        trainer = TrainerReconstruction(DEVICE, dataset, verbose, save, save_aux)
    else:
        trainer = TrainerPrediction(DEVICE, dataset, verbose, save, save_aux)

    if trainer.dataset_name == 'MNIST':
        input_shape = (1, 28, 28)
    elif trainer.dataset_name == 'CIFAR10':
        input_shape = (3, 32, 32)
    else:
        raise ValueError()

    if group_name == 'rotation':
        rotation_group = Group(name=group_name, nr_group_elems=nr_group_elems,
                               base_element=2 * math.pi / nr_group_elems)
        batch_transformer = TransformTensorBatch(kernel=kernel, image_size=torch.Size(
            (input_shape[-2] + 2, input_shape[-1] + 2)),
                                                 device=trainer.device,
                                                 group_sz=rotation_group.nr_group_elems,
                                                 use_scipy_order2=use_scipy_order2)
        transformer = RotationGroupTransformer(group=rotation_group, device=trainer.device,
                                               rotation_batch_tansformer=batch_transformer)
    elif group_name == 'scale':
        scale_group = Group(name=group_name, nr_group_elems=nr_group_elems, base_element=1)
        scale_batch_transformer = TransformTensorBatch(kernel=kernel, image_size=torch.Size(
            (input_shape[-2] + 2, input_shape[-1] + 2)),
                                                       device=trainer.device,
                                                       group_sz=scale_group.nr_group_elems,
                                                       use_scipy_order2=use_scipy_order2)
        transformer = ScaleGroupTransformer(group=scale_group, device=trainer.device,
                                            scale_batch_tansformer=scale_batch_transformer)
    else:
        raise ValueError("No such group")

    config_dict = SimpleNamespace()

    config_dict.use_scipy_order2 = use_scipy_order2
    config_dict.kernel_type = kernel_type
    config_dict.sigma = sigma
    config_dict.width = width
    config_dict.group_name = group_name
    config_dict.nr_group_elems = nr_group_elems
    config_dict.basis_equiv_layers = basis_equiv_layers
    config_dict.fc_sizes = fc_sizes
    config_dict.shape_input = input_shape
    config_dict.sz_output = sz_output
    config_dict.bias = bias
    config_dict.stride_sz_conv = stride_sz_conv
    config_dict.pool_sz_conv = pool_sz_conv
    config_dict.orthogonality_rate = orthg_rate
    config_dict.equivariance_rate = equiv_rate
    config_dict.weight_decay = weight_decay
    config_dict.normalize = normalize
    config_dict.normalized_l2 = normalized_l2
    config_dict.epochs = epochs
    config_dict.lr = lr
    config_dict.dataset = dataset
    config_dict.pool_type = pool_type
    config_dict.last_layer_type = last_layer_type
    config_dict.finetune_batches = finetune_batches
    config_dict.rotation_augmentation_angle = rotation_augmentation_angle
    config_dict.rotation_augmentation_type = rotation_augmentation_type
    config_dict.optimizer = optimizer

    if train_basis_last_epoch is not None:
        assert train_basis_last_epoch <= epochs
    config_dict.train_basis_last_epoch = train_basis_last_epoch
    config_dict.train_basis_every_n_batches = train_basis_every_n_batches
    config_dict.onebyoneconv = onebyoneconv
    config_dict.basis_equiv_layers_type = basis_equiv_layers_type
    model = BasisEquivariantNet(transformer=transformer, basis_equiv_layers=basis_equiv_layers,
                                fc_sizes=fc_sizes, shape_input=input_shape, sz_output=sz_output,
                                bias=bias, stride_conv=stride_sz_conv,
                                pool_sz_conv=pool_sz_conv, normalize_basis=normalize,
                                lr=lr, normalized_l2=normalized_l2, onebyoneconv=onebyoneconv,
                                basis_equiv_layers_type=basis_equiv_layers_type,
                                pool_type=pool_type, last_layer_type=last_layer_type)

    config_dict.load = load
    config_dict.load_aux = load_aux
    assert load in [None, 'basis', 'model']
    if load is not None:
        if load == 'basis':
            trainer.serializer.load_model_basis(model, load_aux, config_dict)
            model.to(DEVICE)
        else:
            trainer.serializer.load_model(model, load_aux, config_dict)
    model = model.to(DEVICE)
    # if torch.cuda.device_count() > 1:  # 0
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = torch.nn.DataParallel(model)
    trainer.set_model(model, config_dict, transformer)
    return trainer, model, transformer, config_dict


def init_train(*args, **kwargs):
    trainer, model, transformer, config_dict = init_trainer_model(*args, **kwargs)
    trainer.train(model=model, transformer=transformer, config_dict=config_dict)
    del model
    torch.cuda.empty_cache()


class TrainerPrediction:
    def __init__(self, device, dataset, verbose, save, save_aux):
        self.dataset_name = dataset
        self.device = device
        self.serializer = Serializer()
        self.target = 'prediction'
        self.trainer_basis = None
        self.writer = None
        self.verbose = verbose
        self.save = save
        self.save_aux = save_aux
        self.model_name = None
        self.config_dict = None
        self.max_accuracy = 0.
        self.transformer = None
        self.epoch_start = time.time()
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.epochs = None
        self.finetune_batches = None
        self.rotation_augmentation_angle = None
        self.rotation_augmentation_type = None

    def set_model(self, model, config_dict, transformer):
        self.epochs = config_dict.epochs
        self.finetune_batches = config_dict.finetune_batches
        self.rotation_augmentation_angle = config_dict.rotation_augmentation_angle
        self.rotation_augmentation_type = config_dict.rotation_augmentation_type
        if self.rotation_augmentation_type == 'torch':
            self.train_loader, self.valid_loader = get_data(dataset=self.dataset_name,
                                                            train_augment_angle=self.rotation_augmentation_angle)
        elif self.rotation_augmentation_type == 'gaussian':
            self.train_loader, self.valid_loader = get_data(dataset=self.dataset_name)

        model_name = get_modelname(config_dict, self.target)
        self.config_dict = config_dict
        self.transformer = transformer
        if os.path.exists('images/' + model_name):
            print("model already trained")
            # return
            # shutil.rmtree('./images/' + model_name)
            # print("overwriting old output file")
            i = 0
            while os.path.exists('images/' + model_name + '_' + str(i)):
                i += 1
            model_name += '_' + str(i)
        path = os.path.join('images', model_name)
        os.mkdir(path)
        if config_dict.last_layer_type != 'group1x1':
            nr_layers = len(config_dict.basis_equiv_layers)
        else:
            nr_layers = len(config_dict.basis_equiv_layers) + 1
        for idx in range(nr_layers):
            os.mkdir(os.path.join(path, 'basis_layer:' + str(idx)))
            os.mkdir(os.path.join(path, 'basis_layer:' + str(idx), 'images'))
            os.mkdir(os.path.join(path, 'layer:' + str(idx)))
            os.mkdir(os.path.join(path, 'layer:' + str(idx), 'images'))

        self.serializer.save_config(model_name, config_dict)

        path_to_net_folder = os.path.join('images', str(model_name))
        self.writer = SummaryWriter(log_dir=path_to_net_folder)

        self.trainer_basis = TrainerBasisAE(config_dict.equivariance_rate, config_dict.orthogonality_rate,
                                            verbose=self.verbose, model_name=model_name,
                                            dataset_len=len(self.train_loader.dataset),
                                            train_loader_len=len(self.train_loader),
                                            log_writer=self.writer,
                                            train_basis_last_epoch=config_dict.train_basis_last_epoch,
                                            train_basis_every_n_batches=config_dict.train_basis_every_n_batches)
        self.model_name = model_name
        print('Device:' + str(DEVICE))
        print('\n\n --model_name created:' + model_name)
        with open(os.path.join(path_to_net_folder, 'model_str.txt'), 'w+') as fd:
            fd.write('Device:' + str(DEVICE))
            fd.write('\n')
            fd.write(str(model))
        print(model)

    def train(self, model, transformer, config_dict=None):

        # self.set_model(model, config_dict, transformer)

        params_except_basis_layers = filter(lambda pair: '.basis.' not in pair[0], model.named_parameters())
        params = [param[1] for param in params_except_basis_layers]
        if self.config_dict.optimizer == 'adam':
            optimizer = torch.optim.Adam(params, lr=config_dict.lr, weight_decay=config_dict.weight_decay, amsgrad=True)
            scheduler = False
        elif self.config_dict.optimizer == 'adam_noams':
            optimizer = torch.optim.Adam(params, lr=config_dict.lr, weight_decay=config_dict.weight_decay, amsgrad=False)
            scheduler = False
        elif self.config_dict.optimizer == 'sgd':
            optimizer = torch.optim.SGD(params, lr=config_dict.lr, momentum=0.9, weight_decay=config_dict.weight_decay)
            milestones = [100, 200]
            scheduler = MultiStepLR(optimizer, milestones, gamma=0.1)
        else:
            raise NotImplementedError

        model = model.to(self.device)

        if self.finetune_batches > 0:
            # TODO change before cluster
            self.validate(model=model, epoch=-2, eq_loss=True)
            self.finetune_basis(model)
        model.freeze_basis()
        self.validate(model=model, epoch=-1, eq_loss=True)

        for epoch in range(0, self.epochs):
            if scheduler:
                scheduler.step(epoch)
            if not OVERFIT_SUBSET:
                self._train_epoch(model=model, optimizer=optimizer, epoch=epoch)
            else:
                self._overfit_small_subset_train(model=model,
                                                 transformer=transformer,
                                                 optimizer=optimizer,
                                                 epoch=epoch,
                                                 verbose=self.verbose)
            # TODO change before cluster
            if epoch != self.epochs - 1:
                self.validate(model=model, epoch=epoch)
            else:
                self.validate(model=model, epoch=epoch+1, eq_loss=True)
                # self.finetune_basis(model)

        model.cpu()

        if self.save:
            if 'model' in self.save_aux:
                self.serializer.save_model(self.model_name, model, config_dict)
                if self.verbose:
                    print("saved model " + self.model_name)
            if 'basis' in self.save_aux and config_dict.epochs <= self.trainer_basis.train_basis_last_epoch:
                self.serializer.save_model_basis(self.model_name, model, config_dict)

        self.model_name = None
        self.config_dict = None
        self.max_accuracy = 0.
        return model

    def validate(self, model, epoch, angle=None, eq_loss=False):

        model.eval()
        loss_sum = 0.
        correct = 0

        if epoch < self.trainer_basis.train_basis_last_epoch:
            self.trainer_basis.reset_log_sums(model.len_non1_basis_equiv_layers)

        # 1 Compute validation
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_loader):
                data = data.to(self.device)
                # data = data.flip([-2, -1])  # 180*
                # data = data.transpose(-2, -1).flip(-1)  # 90*
                # data = data.transpose(-2, -1).flip(-2)  # 270*
                target = target.to(self.device)
                if angle is not None:
                    data = self.transformer.apply_rotation_to_input(data, [angle])

                output, _ = model.forward_prediction(data, epoch=epoch, batch_idx=batch_idx,
                                                     trainer_basis=self.trainer_basis, writer=self.writer,
                                                     dataset_len=len(self.train_loader.dataset))
                loss = self.criterion(output, target)
                loss_sum += loss.item()
                pred = output.argmax(1)
                correct += pred.eq(target).sum().cpu().item()

            if epoch < self.trainer_basis.train_basis_last_epoch:
                rec_loss_sum, rec_loss_norm_sum, equiv_loss_sum, equiv_loss_norm_sum = self.trainer_basis.get_log_sums()
                rec_loss_sum = [x / len(self.valid_loader) for x in rec_loss_sum]
                rec_loss_norm_sum = [x / len(self.valid_loader) for x in rec_loss_norm_sum]
                equiv_loss_sum = [x / len(self.valid_loader) for x in equiv_loss_sum]
                equiv_loss_norm_sum = [x / len(self.valid_loader) for x in equiv_loss_norm_sum]
                for layer_idx in range(len(rec_loss_sum)):
                    str_output = '[Validation: Layer:{} Epoch:{}]:\tRec Loss per pixel:{:.6f}\t Norm Rec Loss per ' \
                                 'image:{'':.6f} \tEquiv Loss per pixel:{:.6f}\tNorm Equiv Loss per image:{' \
                                 ':.6f}'.format(layer_idx, epoch,
                                                rec_loss_sum[layer_idx].item(),
                                                rec_loss_norm_sum[layer_idx].item(),
                                                equiv_loss_sum[layer_idx].item(),
                                                equiv_loss_norm_sum[layer_idx].item(), )
                    if self.verbose:
                        print(str_output)
                    path_to_layer_folder = os.path.join('images', self.model_name, 'basis_layer:' + str(layer_idx))

                    output_file = os.path.join(path_to_layer_folder, 'output.txt')

                    with open(output_file, "a+") as f:
                        f.write(str_output)
                        f.write("\n")

                    # TODO log every epoch statistics here
                    self.writer.add_scalar('basis_layer:' + str(layer_idx) + '/stats/val/reconstrution_loss',
                                            rec_loss_sum[layer_idx],
                                            global_step=len(self.train_loader.dataset) * (epoch + 1))
                    self.writer.add_scalar('basis_layer:' + str(layer_idx) + '/stats/val/reconstruction_loss_norm',
                                            rec_loss_norm_sum[layer_idx],
                                            global_step=len(self.train_loader.dataset) * (epoch + 1))
                    self.writer.add_scalar('basis_layer:' + str(layer_idx) + '/stats/val/equivariance_loss',
                                            equiv_loss_sum[layer_idx],
                                            global_step=len(self.train_loader.dataset) * (epoch + 1))
                    self.writer.add_scalar('basis_layer:' + str(layer_idx) + '/stats/val/equivariance_loss_norm',
                                            equiv_loss_norm_sum[layer_idx],
                                            global_step=len(self.train_loader.dataset) * (epoch + 1))

            loss_sum /= len(self.valid_loader)

            accuracy = float(correct) / len(self.valid_loader.dataset)
            accuracy *= 100
            str_output = '[Validation: Epoch:{}]:\tAccuracy:{:.6f}\t NLL:{'':.6f}\t duration:'.format(epoch, accuracy, loss_sum)
            str_output += str(time.time() - self.epoch_start)
            self.epoch_start = time.time()

            if self.verbose:
                print(str_output)
            path_to_model_output = os.path.join('images', self.model_name, 'output.txt')
            with open(path_to_model_output, "a+") as f:
                f.write(str_output)
                f.write("\n")

            # TODO log every epoch statistics here
            self.writer.add_scalar('stats/val/accuracy',
                                   accuracy,
                                   global_step=len(self.train_loader.dataset) * (epoch + 1))
            self.writer.add_scalar('stats/val/NLL',
                                   loss_sum,
                                   global_step=len(self.train_loader.dataset) * (epoch + 1))

            if accuracy > self.max_accuracy:
                if self.save and 'model' in self.save_aux:
                    if self.trainer_basis.train_basis_last_epoch is None or \
                            epoch >= self.trainer_basis.train_basis_last_epoch:
                        self.serializer.save_model(model_name=self.model_name, model=model,
                                                   config_dict=self.config_dict, aux='best', acc=accuracy)
                self.max_accuracy = accuracy

        # 2 compute equivariance images, basis_plot, orthogonolaity_plot
        if (epoch+1) % 10 == 0 or epoch < 1:
            with torch.no_grad():
                data_shape_len = len(data.shape)
                rand_idx = np.random.randint(0, data.shape[0])
                one_image_subset = data[[rand_idx]].repeat(BATCH_SIZE, *[1] * (data_shape_len - 1))
                tranformation_indices = torch.tensor(
                    [i % self.transformer.group.nr_group_elems for i in range(BATCH_SIZE)], dtype=torch.float,
                    device=data.device)
                self.transformer.set_elements_sample(tranformation_indices)
                one_image_rotated = self.transformer.apply_sample_action_to_input(one_image_subset)

                model.forward_prediction(input=one_image_rotated, epoch=epoch, batch_idx=-1, trainer_basis=self.trainer_basis)

        if eq_loss:
            # 3 compute actual equivariance loss
            self.trainer_basis.reset_l2_normalized_total_equivariance_total_sums(len(model.basis_equiv_layers))
            with torch.no_grad():
                data_shape_len = len(data.shape)
                for batch_idx, (data, target) in enumerate(self.valid_loader):
                    data = data.to(self.device)
                # for idx in range(data.shape[0]):
                    one_image_subset = data[[0]].repeat(BATCH_SIZE, *[1] * (data_shape_len - 1))
                    tranformation_indices = torch.tensor(
                        [i % self.transformer.group.nr_group_elems for i in range(BATCH_SIZE)], dtype=torch.float,
                        device=data.device)
                    self.transformer.set_elements_sample(tranformation_indices)
                    one_image_rotated = self.transformer.apply_sample_action_to_input(one_image_subset)

                    # bnorm_adapted_input
                    # because only the first 8 images are important when computing the eq loss
                    one_image_rotated[self.transformer.group.nr_group_elems:] = data[self.transformer.group.nr_group_elems:]
                    model.forward_prediction(input=one_image_rotated, epoch=epoch, batch_idx=batch_idx,
                                             trainer_basis=self.trainer_basis, eq_loss=True)
                    if batch_idx == 100:
                        break

                for layer in model.layers:
                    if type(layer) == BasisEquivConvLyer and layer.filter_sz != 1:
                        path_to_layer_folder = os.path.join('images', self.model_name, 'basis_layer:' + str(layer.index))
                        path_to_layer_images = os.path.join(path_to_layer_folder, 'images')
                        fig_number = 'epoch:' + str(epoch) + '_batch:' + str(-2) + '_' + str(layer.index) + '_'
                        layer.basis_ae.basis.plot(fig_name=fig_number, path_to_layer_images=path_to_layer_images)

            per_layer_eq = self.trainer_basis.get_l2_normalized_total_equivariance_total_sums()
            for idx, layer_eq in enumerate(per_layer_eq):
                path_to_layer_folder = os.path.join('images', self.model_name, 'basis_layer:' + str(idx))
                with open(os.path.join(path_to_layer_folder, 'output.txt'), 'a+') as fd:
                    fd.write('Layer FULL equivariance: '+str(layer_eq.item()))
                    fd.write('\n')
                if self.verbose:
                    print('Layer '+str(idx)+' FULL equivariance: '+str(layer_eq.item()))
                self.writer.add_scalar('basis_layer:' + str(idx) + '/stats/val/full_equivariance',
                                       layer_eq.item(), global_step=len(self.train_loader.dataset) * (epoch+1))

            path_to_model_output = os.path.join('images', self.model_name, 'output.txt')
            with open(path_to_model_output, 'a+') as fd:
                fd.write('Model FULL equivariance: ' + ', '.join([str(i.item()) for i in per_layer_eq]))
                fd.write('\n')

        if epoch + 1 == self.trainer_basis.train_basis_last_epoch:
            if self.save and 'basis' in self.save_aux:
                model = model.to(torch.device('cpu'))
                self.serializer.save_model_basis(self.model_name, model, self.config_dict)
                model = model.to(DEVICE)
            model.freeze_basis()

        # TODO remove this
        # with open(path_to_model_output, "a+") as f:
        #     f.write('extra time in validation: '+str(time.time() - self.epoch_start))
        #     f.write("\n")
        return loss_sum, accuracy

    def _train_epoch(self, model, optimizer, epoch):

        optimizer.zero_grad()
        start = time.time()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            model.train()

            data = data.to(self.device)
            target = target.to(self.device)

            if self.rotation_augmentation_type == 'gaussian': # and self.rotation_augmentation_angle != '0'
                with torch.no_grad():
                    if self.rotation_augmentation_angle == 'all':
                        angles = np.random.rand(BATCH_SIZE)*math.pi
                    elif self.rotation_augmentation_angle == '0':
                        angles = np.zeros(BATCH_SIZE)
                    else:
                        t_indices = np.random.randint(0, int(360 / int(self.rotation_augmentation_angle)), BATCH_SIZE)
                        base_element = float(2 * math.pi) / 360. * int(self.rotation_augmentation_angle)
                        angles = base_element * t_indices

                    data = self.transformer.apply_rotation_to_input(data, angles)

            output, ae_loss = model.forward_prediction(input=data, epoch=epoch, batch_idx=batch_idx,
                                                       trainer_basis=self.trainer_basis)

            acc_loss = self.criterion(output, target)
            loss = acc_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (batch_idx == 0 and epoch == 0) or (batch_idx + 1) % LOG_INTERVAL == 0:
                str_output = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tNLL: {:.6f}\t duration:'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader), acc_loss.item())
                str_output += str(time.time() - start)
                start = time.time()
                if self.verbose:
                    print(str_output)
                path_to_layer_output = os.path.join('images', self.model_name, 'output.txt')
                with open(path_to_layer_output, "a+") as f:
                    f.write(str_output)
                    f.write("\n")
            # TODO log train pred loss every step
            self.writer.add_scalar('stats/train/NLL',
                                   acc_loss.item(),
                                   global_step=len(self.train_loader.dataset) * (epoch) + (batch_idx + 1) * BATCH_SIZE)

            # TODO remove this
            # if (batch_idx == 0 and epoch == 0) or (batch_idx + 1) % LOG_INTERVAL == 0:
            #     with open(path_to_layer_output, "a+") as f:
            #         f.write('extra time in train: ' + str(time.time() - start))
            #         f.write("\n")

    def finetune_basis(self, model):
        model.train()
        for layer in model.layers:
            # ...
            if type(layer) == BasisEquivConvLyer and layer.filter_sz != 1:
                layer.basis_ae.basis.FROZEN = False
                layer.basis_ae.basis._unnormalized_basis.requires_grad = True

        params_basis_layers = filter(lambda pair: '.basis.' in pair[0], model.named_parameters())
        params = [param[1] for param in params_basis_layers]
        optimizer = torch.optim.Adam(params, lr=1e-3, amsgrad=True)


        count_0_angle_images = int(BATCH_SIZE / self.transformer.group.nr_group_elems)
        angle_0_indices = torch.tensor([i * self.transformer.group.nr_group_elems for i in range(count_0_angle_images)], dtype=torch.long)
        range_after_0 = torch.tensor([i for i in range(1, self.transformer.group.nr_group_elems)], dtype=torch.long)
        indices_angles_after_0 = angle_0_indices[:, None] + range_after_0[None, :]

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device)
            target = target.to(self.device)

            data_shape_len = len(data.shape)
            # select BATCH_SIZE/group_elems random indices
            rand_indices = np.random.randint(0, data.shape[0], count_0_angle_images+1)
            # select subset of images and create new axis to repeat images for each group element
            images_subset = data[rand_indices][:, None]
            images_subset = images_subset.repeat(1, self.transformer.group.nr_group_elems, *[1] * (data_shape_len - 1))
            images_subset = images_subset.view(-1, *images_subset.shape[2:])[0:BATCH_SIZE]
            tranformation_indices = torch.tensor(
                [i % self.transformer.group.nr_group_elems for i in range(BATCH_SIZE)], dtype=torch.float,
                device=data.device)
            self.transformer.set_elements_sample(tranformation_indices)
            rotated_subset = self.transformer.apply_sample_action_to_input(images_subset)

            activations = model.forward_prediction(input=rotated_subset, epoch=-1, batch_idx=batch_idx,
                                                   trainer_basis=None, finetune_basis=True)

            backward_indices = -tranformation_indices % self.transformer.group.nr_group_elems
            rolled_back_y = self.transformer.apply_roll(activations, 2, backward_indices, 0)
            rolled_rotated_back_y = self.transformer.apply_sample_action_to_input(rolled_back_y,
                                                                                  backward_indices)

            images_at_angle_0 = rolled_rotated_back_y[angle_0_indices][:, None].repeat(1, 7, *[1 for _ in range(len(rolled_rotated_back_y.shape)-1)])
            images_after_angle_0 = rolled_rotated_back_y[indices_angles_after_0.view(-1)].view(count_0_angle_images, 7, *rolled_rotated_back_y.shape[1:])

            images_after_angle_0 = images_after_angle_0.view(images_after_angle_0.shape[0]*images_after_angle_0.shape[1], *images_after_angle_0.shape[2:])
            images_at_angle_0 = images_at_angle_0.view(images_at_angle_0.shape[0] * images_at_angle_0.shape[1], *images_at_angle_0.shape[2:])
            l2_per_pixel, normalized_equiv_error, _ = Loss.get_normalized_l2_loss_at_non_zero_indices(images_after_angle_0, images_at_angle_0, normalized_l2=True)

            # images_at_angle_0 = rolled_rotated_back_y[angle_0_indices][:, None]
            # images_after_angle_0 = rolled_rotated_back_y[indices_angles_after_0.view(-1)].view(count_0_angle_images,
            #                                                                                    self.transformer.group.nr_group_elems - 1,
            #                                                                                    *rolled_rotated_back_y.shape[
            #                                                                                     1:])
            # error = (images_at_angle_0.detach() - images_after_angle_0)
            # error = error.pow(2).view(error.shape[0], 7, -1).sum(-1)
            # error_norm = images_at_angle_0.pow(2).view(images_at_angle_0.shape[0], 1, -1).sum(-1).sqrt() * \
            #              images_after_angle_0.pow(2).view(images_after_angle_0.shape[0], 7, -1).sum(-1).sqrt()
            # normalized_equiv_error = error / error_norm
            # normalized_equiv_error = normalized_equiv_error.mean()

            optimizer.zero_grad()
            normalized_equiv_error.backward()
            optimizer.step()

            self.writer.add_scalar('stats/finetune/loss',
                                   normalized_equiv_error.item(),
                                   global_step=len(self.train_loader.dataset) * (0) + (batch_idx + 1) * BATCH_SIZE)

            if batch_idx == self.finetune_batches-1:
                print('Last finetune loss: '+str(normalized_equiv_error.item()))
                break

    def _overfit_small_subset_train(self, model, transformer, optimizer, epoch, criterion,
                                    verbose):
        raise NotImplementedError
        model.train()

        if epoch == 0:
            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.batch_idx = 0
                self.data = data
                self.target = target
                break

        acc_loss_vector, equiv_loss_vector, orthg_loss_vector = [], [], []

        optimizer.zero_grad()
        data = self.data.to(self.device)
        target = self.target.to(self.device)

        transformer.set_elements_sample()
        rot_data = transformer.apply_sample_action_to_input(data)
        output = model(data, rot_data)

        # acc_loss = F.nll_loss(output, target)
        acc_loss = criterion(output, target)
        equiv_loss, orthg_loss = model.get_and_reset_loss()

        acc_loss_vector.append(acc_loss.item() if acc_loss is not None else None)
        equiv_loss_vector.append(equiv_loss.item() if equiv_loss is not None else None)
        orthg_loss_vector.append(orthg_loss.item() if orthg_loss is not None else None)

        # loss = equiv_loss
        loss = acc_loss
        loss = torch.add(loss, equiv_loss)
        loss = torch.add(loss, orthg_loss)

        loss.backward()
        optimizer.step()
        if verbose and ((self.batch_idx == 0 and epoch == 0) or (self.batch_idx + 1) % LOG_INTERVAL == 0):
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tNLL: {:.6f}\tEquiv Loss: {:.6f}\tOrthg Loss: {:.6f}'.format(
                    epoch, self.batch_idx * len(data), len(self.train_loader.dataset),
                           100. * self.batch_idx / len(self.train_loader), acc_loss.item(), equiv_loss.item(),
                    orthg_loss.item()))

        return acc_loss_vector, equiv_loss_vector, orthg_loss_vector


class TrainerReconstruction:
    def __init__(self, device, dataset, verbose, save, save_aux):
        self.dataset_name = dataset
        self.train_loader, self.valid_loader = get_data(dataset=dataset)
        self.device = device
        self.serializer = Serializer()
        self.target = 'reconstruction'
        self.trainer_basis = None
        self.writer = None
        self.verbose = verbose
        self.save = save
        self.save_aux = save_aux
        self.model_name = None
        self.config_dict = None
        self.transformer = None

    def set_model(self, model, config_dict, transformer):
        model_name = get_modelname(config_dict, self.target)
        self.config_dict = config_dict
        self.transformer = transformer
        if os.path.exists('images/' + model_name):
            print("model already trained")
            # return
            # shutil.rmtree('./images/' + model_name)
            # print("overwriting old output file")
            i = 0
            while os.path.exists('images/' + model_name + '_'+str(i)):
                i += 1
            model_name += '_' + str(i)
        path = os.path.join('images', model_name)
        os.mkdir(path)
        for idx in range(len(config_dict.basis_equiv_layers)):
            os.mkdir(os.path.join(path, 'basis_layer:' + str(idx)))
            os.mkdir(os.path.join(path, 'basis_layer:' + str(idx), 'images'))
            os.mkdir(os.path.join(path, 'layer:' + str(idx)))
            os.mkdir(os.path.join(path, 'layer:' + str(idx), 'images'))

        self.serializer.save_config(model_name, config_dict)

        path_to_net_folder = os.path.join('images', str(model_name))
        self.writer = SummaryWriter(log_dir=path_to_net_folder)

        self.trainer_basis = TrainerBasisAE(config_dict.equivariance_rate, config_dict.orthogonality_rate,
                                            verbose=self.verbose, model_name=model_name,
                                            dataset_len=len(self.train_loader.dataset),
                                            train_loader_len=len(self.train_loader),
                                            log_writer=self.writer)
        self.model_name = model_name
        print('Device:' + str(DEVICE))
        print('\n\n --model_name created:' + model_name)
        with open(os.path.join(path_to_net_folder, 'model_str.txt'), 'w+') as fd:
            fd.write('Device:' + str(DEVICE))
            fd.write('\n')
            fd.write(model_name)
        print(model)

    def train(self, model, transformer, config_dict):
        # self.set_model(model, config_dict, transformer)

        model = model.to(self.device)

        self.validate(model, epoch=-1)

        # Train
        for epoch in range(0, config_dict.epochs):
            start = time.time()
            self._train_epoch(model=model, epoch=epoch)
            self.validate(model, epoch)
            end = time.time()
            if self.verbose:
                print("epoch duration in seconds: " + str(end - start))
        model.to(torch.device('cpu'))

        if self.save and 'basis' in self.save_aux:
            self.serializer.save_model_basis(self.model_name, model, config_dict)
            if self.verbose:
                print("saved model " + self.model_name)

        self.model_name = None
        self.config_dict = None
        return model

    def _train_epoch(self, model, epoch):

        for batch_idx, (data, target) in enumerate(self.train_loader):
            model.train()

            data = data.to(self.device)
            model.forward_reconstruction(input=data, epoch=epoch, batch_idx=batch_idx, trainer_basis=self.trainer_basis)

    def validate(self, model, epoch):
        model.eval()
        self.trainer_basis.reset_log_sums(model.len_non1_basis_equiv_layers)
        # 1 compute validation
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_loader):
                data = data.to(self.device)
                # target = target.to(self.device)

                with torch.no_grad():
                    _ = model.forward_reconstruction(data, epoch=epoch, batch_idx=batch_idx,
                                                     trainer_basis=self.trainer_basis)

            rec_loss_sum, rec_loss_norm_sum, equiv_loss_sum, equiv_loss_norm_sum = self.trainer_basis.get_log_sums()
            rec_loss_sum = [x / len(self.valid_loader) for x in rec_loss_sum]
            rec_loss_norm_sum = [x / len(self.valid_loader) for x in rec_loss_norm_sum]
            equiv_loss_sum = [x / len(self.valid_loader) for x in equiv_loss_sum]
            equiv_loss_norm_sum = [x / len(self.valid_loader) for x in equiv_loss_norm_sum]

            for layer_idx in range(len(rec_loss_sum)):
                str_output = '[Validation: Layer:{} Epoch:{}]:\tRec Loss per pixel:{:.6f}\t Norm Rec Loss per ' \
                             'image:{'':.6f} \tEquiv Loss per pixel:{:.6f}\tNorm Equiv Loss per image:{' \
                             ':.6f}'.format(layer_idx, epoch,
                                            rec_loss_sum[layer_idx].item(),
                                            rec_loss_norm_sum[layer_idx].item(),
                                            equiv_loss_sum[layer_idx].item(),
                                            equiv_loss_norm_sum[layer_idx].item(), )
                if self.verbose:
                    print(str_output)
                path_to_layer_folder = os.path.join('images', self.model_name, 'basis_layer:' + str(layer_idx))

                output_file = os.path.join(path_to_layer_folder, 'output.txt')

                with open(output_file, "a+") as f:
                    f.write(str_output)
                    f.write("\n")

                # TODO log every epoch statistics here
                self.writer.add_scalar('basis_layer:' + str(layer_idx) + '/stats/val/reconstrution_loss',
                                        rec_loss_sum[layer_idx],
                                        global_step=len(self.train_loader.dataset) * (epoch + 1))
                self.writer.add_scalar('basis_layer:' + str(layer_idx) + '/stats/val/reconstruction_loss_norm',
                                        rec_loss_norm_sum[layer_idx],
                                        global_step=len(self.train_loader.dataset) * (epoch + 1))
                self.writer.add_scalar('basis_layer:' + str(layer_idx) + '/stats/val/equivariance_loss',
                                        equiv_loss_sum[layer_idx],
                                        global_step=len(self.train_loader.dataset) * (epoch + 1))
                self.writer.add_scalar('basis_layer:' + str(layer_idx) + '/stats/val/equivariance_loss_norm',
                                        equiv_loss_norm_sum[layer_idx],
                                        global_step=len(self.train_loader.dataset) * (epoch + 1))

        # 2 compute equivariance images, basis_plot, orthogonolaity_plot
        with torch.no_grad():
            data_shape_len = len(data.shape)
            rand_idx = np.random.randint(0, data.shape[0])
            one_image_subset = data[[rand_idx]].repeat(BATCH_SIZE, *[1] * (data_shape_len - 1))
            tranformation_indices = torch.tensor(
                [i % self.transformer.group.nr_group_elems for i in range(BATCH_SIZE)], dtype=torch.float,
                device=data.device)
            self.transformer.set_elements_sample(tranformation_indices)
            data = self.transformer.apply_sample_action_to_input(one_image_subset)

            model.forward_reconstruction(input=data, epoch=epoch, batch_idx=-1, trainer_basis=self.trainer_basis)

    def load_from_basis(self, model, list_paths_to_basis):
        self.serializer.load_model_basis(model, list_paths_to_basis)
        model = model.to(self.device)
        return model
