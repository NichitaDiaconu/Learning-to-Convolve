import sys

from constants import DEVICE
from trainer import TrainerPrediction, init_train, TrainerReconstruction
import numpy as np

dataset = 'CIFAR10'

weight_decay = 1e-6

bias = True

verbose = True

stride_sz_conv = [2]  # , 2]
basis_equiv_layers = [None]
# sigma = 0.5  # 0.5
nr_filters = 5
kernel_width = 3
lr = 0.003

target = 'reconstruction'
epochs = 10

save = True
save_aux = ['basis', 'model']
load = None
basis_equiv_layers_type = 'learned'


# for equiv_rate, orthg_rate in [(30, 1), (10, 1)]:  # , (30, 1), (10, 3), (30, 3)]:
#     for sigma in [0.5, 0.3]:  # , 0.2, 0.7]:
#         for filter_sz in [5, 7]:
equiv_rate = 1  # float(sys.argv[1])
orthg_rate = 1  # sys.argv[-1]
sigma = 0.3  # float(sys.argv[2])
filter_sz = 3  # int(sys.argv[3])

normalized_l2 = True
normalize_basis = None  # , (None, 'Experiment 1'), (None, 'Experiment 3')]:  # , [45, False, True, False, True], (None, 'Experiment 4'), (None, 'Experiment 3')]:  # (None, 'Experiment 2'), # [[45, False, True, False, True]]:  # , None, [25, True, True, True, True]
nr_basis = filter_sz ** 2
layer = (nr_basis, nr_filters, filter_sz)
basis_equiv_layers[0] = layer

load_aux = None
onebyoneconv = []
pool_type = 'stride'
last_layer_type = 'linear'
finetune_batches = 0

init_train(lr=lr, epochs=epochs, stride_sz_conv=stride_sz_conv,
           basis_equiv_layers=basis_equiv_layers,
           kernel_type='Gaussian', width=kernel_width, sigma=sigma,
           use_scipy_order2=False, save=True,
           equiv_rate=equiv_rate, orthg_rate=orthg_rate, weight_decay=weight_decay,
           normalize=normalize_basis, verbose=verbose,
           train_basis_every_n_batches=None,
           train_basis_last_epoch=None, bias=False,
           group_name='rotation', nr_group_elems=8,
           fc_sizes=[10], normalized_l2=normalized_l2, dataset=dataset, target=target,
           save_aux=save_aux, load=load, basis_equiv_layers_type=basis_equiv_layers_type, load_aux=load_aux, onebyoneconv=onebyoneconv, pool_type=pool_type, last_layer_type=last_layer_type, finetune_batches=finetune_batches)
