from basis_equivariant_network import BasisEquivariantNet
from constants import DEVICE
from trainer import TrainerPrediction, init_train, TrainerReconstruction, init_trainer_model
import numpy as np

dataset = 'CIFAR10'

weight_decay = 1e-6
train_basis_last_epoch = 1
train_basis_every_n_batches = 1
bias = True

stride_sz_conv = [2, 2]
basis_equiv_layers = [None, None]
layer = (9, 8, 3)
kernel_width = 3

target = 'reconstruction'
normalized_l2 = None
basis_equiv_layers[0] = layer
basis_equiv_layers[1] = layer
epochs = 1
fc_sizes = [2048, 1024, 100, 100]
sigma = 0.3
verbose = True
save = False
save_aux = 'basis'

load = 'basis'
path_to_layer = './images/LEGACY/reconstruction_group_name:rotation sigma:0.3 equivariance_rate:10 orthogonality_rate:1 normalize:None normalized_l2:False l2_coef:1e-06/basis_layer:0'
load_aux = [path_to_layer, path_to_layer]
normalize = None
trainer, model, transformer, config_dict = init_trainer_model(lr=0.003, epochs=epochs, stride_sz_conv=stride_sz_conv,
                                                              basis_equiv_layers=basis_equiv_layers,
                                                              kernel_type='Gaussian', width=kernel_width, sigma=sigma,
                                                              use_scipy_order2=False, save=save,
                                                              equiv_rate=None, orthg_rate=None,
                                                              weight_decay=weight_decay,
                                                              normalize=normalize, verbose=verbose,
                                                              train_basis_every_n_batches=None,
                                                              train_basis_last_epoch=None, bias=False,
                                                              group_name='rotation', nr_group_elems=8,
                                                              fc_sizes=[2048, 1024, 100, 100],
                                                              normalized_l2=normalized_l2, dataset=dataset,
                                                              target=target, load=load, load_aux=load_aux)
trainer.set_model(model, config_dict, transformer)
trainer.validate(model, -1)
