from trainer import TrainerPrediction, init_train, TrainerReconstruction, init_trainer_model
import sys
dataset = 'MNIST'

train_basis_last_epoch = -2
train_basis_every_n_batches = 1
bias = True

weight_decay = 1e-6
# stride_sz_conv = [1, 1, 2, 1, 1, 2, 1]
# basis_equiv_layers = [(9, 33, 3), (9, 33, 3), (9, 33, 3), (9, 67, 3), (9, 67, 3), (9, 67, 3), (9, 67, 3)]
stride_sz_conv = [1, 2, 1, 2]  # , 1, 1, 2, 1, 1, 2, 1, 1]
basis_equiv_layers = [(9, 22, 3), (9, 22, 3), (9, 44, 3), (9, 44, 3)]  #, (9, 88, 3), (9, 88, 3), (9, 88, 3), (9, 176, 3), (9, 176, 3), (9, 176, 3), (9, 176, 3), (9, 176, 3), (9, 176, 3)]
basis_equiv_layers.extend([(0, 44, 1), (0, 44, 1)])
stride_sz_conv.extend([1, 1])
lr = 1e-3  # 0.1
# weight_decay = 1e-3

kernel_width = 3

target = 'prediction'
normalized_l2 = None
epochs = 1  # 100
fc_sizes = []
onebyoneconv = []
sigma = 0.5
verbose = True
save = True
save_aux = ['basis', 'model']  # 'basis'

load = 'basis'
# path_to_layer_basis = sys.argv[1]
path_to_layer_basis = "./images/LEGACY/reconstruction_group_name:rotation sigma:0.5 equivariance_rate:30 orthogonality_rate:1 normalize:None normalized_l2:False l2_coef:1e-06/basis_layer:0"
load_aux = [path_to_layer_basis] * len(basis_equiv_layers)
normalize = None

finetune_batches = 300
last_layer_type = 'linear'
pool_type = 'max'
basis_equiv_layers_type = 'learned'


init_train(lr=lr, epochs=epochs, stride_sz_conv=stride_sz_conv,
           basis_equiv_layers=basis_equiv_layers,
           kernel_type='Gaussian', width=kernel_width, sigma=sigma,
           use_scipy_order2=False, save=save, save_aux=save_aux,
           equiv_rate=None, orthg_rate=None,
           weight_decay=weight_decay,
           normalize=normalize, verbose=verbose,
           train_basis_every_n_batches=train_basis_every_n_batches,
           train_basis_last_epoch=train_basis_last_epoch, bias=bias,
           group_name='rotation', nr_group_elems=8,
           fc_sizes=fc_sizes,
           normalized_l2=normalized_l2, dataset=dataset,
           target=target, load=load, load_aux=load_aux, onebyoneconv=onebyoneconv,
           finetune_batches=finetune_batches, last_layer_type=last_layer_type, pool_type=pool_type,
           basis_equiv_layers_type=basis_equiv_layers_type)
