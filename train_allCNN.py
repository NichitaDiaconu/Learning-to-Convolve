import sys

from trainer import TrainerPrediction, init_train, TrainerReconstruction, init_trainer_model

# TODO change this before cluster
dataset = sys.argv[1]  # 'MNIST' #
basis_equiv_layers_type = sys.argv[2]  # 'average' # 'overcomplete'
pool_type = sys.argv[3]  # 'max'  #
last_layer_type = sys.argv[4]  # 'linear'  #
finetune_batches = int(sys.argv[5])  # 100  #
rotation_augmentation_angle = sys.argv[6]
rotation_augmentation_type = 'gaussian'
sigma = float(sys.argv[9])  # 0.35

optimizer = sys.argv[7]
if optimizer == 'sgd':
    lr = 1e-1
else:
    lr = 1e-3

if basis_equiv_layers_type in ['learned', 'average', 'gaussian', 'bilinear']:
    load_basis_type = sys.argv[8]  # (10,1)
else:
    load_basis_type = None
# dataset = 'CIFAR10' #
# basis_equiv_layers_type = 'gaussian' #
# pool_type = 'max'  #
# last_layer_type = 'linear'  #
# finetune_batches = 100  #
# load_basis_type = '(10,1)'

train_basis_every_n_batches = 1
bias = True
weight_decay = 1e-6

kernel_width = 3
target = 'prediction'
normalized_l2 = None
epochs = 100
verbose = True
save = True  # True
save_aux = ['basis', 'model']
equiv_rate = None  # 10
orthg_rate = None  # 1
normalize = None
# finetune_batches= 0  #
# TODO change this before cluster. to -2
train_basis_last_epoch = -2

if basis_equiv_layers_type == 'conv':
    stride_sz_conv = [1, 1, 2, 1, 1, 2, 1]
    basis_equiv_layers = [(None, 96, 3), (None, 96, 3), (None, 96, 3), (None, 192, 3), (None, 192, 3), (None, 192, 3), (None, 192, 3)]
    onebyoneconv = [192, 192]
    load = None
    load_aux = None
    fc_sizes = []

elif basis_equiv_layers_type == 'weiler':
    stride_sz_conv = [1, 1, 2, 1, 1, 2, 1]
    basis_equiv_layers = [(9, 33, 3), (9, 33, 3), (9, 33, 3), (9, 67, 3), (9, 67, 3), (9, 67, 3), (9, 67, 3)]
    if last_layer_type == 'conv1x1':
        onebyoneconv = [192, 192]
    elif last_layer_type == 'linear':
        basis_equiv_layers.extend([(0, 67, 1), (0, 67, 1)])
        stride_sz_conv.extend([1, 1])
        onebyoneconv = []
    elif last_layer_type == 'group1x1':
        basis_equiv_layers.extend([(0, 67, 1), (0, 67, 1)])
        stride_sz_conv.extend([1, 1])
        onebyoneconv = []
    load = None
    load_aux = None
    fc_sizes = []

elif basis_equiv_layers_type == 'random':
    stride_sz_conv = [1, 1, 2, 1, 1, 2, 1]
    basis_equiv_layers = [(9, 33, 3), (9, 33, 3), (9, 33, 3), (9, 67, 3), (9, 67, 3), (9, 67, 3), (9, 67, 3)]
    if last_layer_type == 'conv1x1':
        onebyoneconv = [192, 192]
    elif last_layer_type == 'linear':
        basis_equiv_layers.extend([(0, 67, 1), (0, 67, 1)])
        stride_sz_conv.extend([1, 1])
        onebyoneconv = []
    elif last_layer_type == 'group1x1':
        basis_equiv_layers.extend([(0, 67, 1), (0, 67, 1)])
        stride_sz_conv.extend([1, 1])
        onebyoneconv = []
    load = None
    load_aux = None
    fc_sizes = []

elif basis_equiv_layers_type in ['learned', 'average', 'gaussian', 'bilinear']:
    # finetune_batches= 100
    stride_sz_conv = [1, 1, 2, 1, 1, 2, 1]
    basis_equiv_layers = [(9, 33, 3), (9, 33, 3), (9, 33, 3), (9, 67, 3), (9, 67, 3), (9, 67, 3), (9, 67, 3)]
    load = 'basis'

    if load_basis_type == 'None':
        path_to_layer = './images/LEGACY/reconstruction_group_name:rotation sigma:0.5 equivariance_rate:30 orthogonality_rate:1 normalize:None normalized_l2:False l2_coef:1e-06/basis_layer:0'
    if load_basis_type == '(10,3)':
        # Experiment 1 10, 3
        path_to_layer = 'images/LEGACY/reconstruction_group_name:rotation sigma:0.3 equivariance_rate:10 orthogonality_rate:3 normalize:(None,\'Experiment1\') normalized_l2:False l2_coef:1e-06/basis_layer:0'
    if load_basis_type == '(10,1)':
        # Experiment 1 10, 1
        path_to_layer = 'images/LEGACY/reconstruction_group_name:rotation sigma:0.3 equivariance_rate:10 orthogonality_rate:1 normalize:(None,\'Experiment1\') normalized_l2:False l2_coef:1e-06/basis_layer:0'

    load_aux = [path_to_layer] * (len(basis_equiv_layers))
    if last_layer_type == 'conv1x1':
        onebyoneconv = [192, 192]
    elif last_layer_type == 'linear':
        basis_equiv_layers.extend([(0, 67, 1), (0, 67, 1)])
        stride_sz_conv.extend([1, 1])
        onebyoneconv = []
        # onebyoneconv = [192, 192]
    elif last_layer_type == 'group1x1':
        basis_equiv_layers.extend([(0, 67, 1), (0, 67, 1)])
        stride_sz_conv.extend([1, 1])
        onebyoneconv = []
    fc_sizes = []
elif basis_equiv_layers_type == 'overcomplete':
    stride_sz_conv = [1, 1, 2, 1, 1, 2, 1]
    # basis_equiv_layers = [(3*9, 33, 3), (3*9, 33, 3), (3*9, 33, 3), (3*9, 67, 3), (3*9, 67, 3), (3*9, 67, 3), (3*9, 67, 3)]
    basis_equiv_layers = [(6*9, 33, 3), (6*9, 33, 3), (6*9, 33, 3), (6*9, 67, 3), (6*9, 67, 3), (6*9, 67, 3), (6*9, 67, 3)]
    load = 'basis'
    basis_equiv_layers_type = 'average'
    # None; Experiment 1 10, 1; Experiment 1 10, 3
    path_to_layer = ('./images/LEGACY/reconstruction_group_name:rotation sigma:0.3 equivariance_rate:30 orthogonality_rate:1 normalize:None normalized_l2:False l2_coef:1e-06/basis_layer:0',
                     './images/LEGACY/reconstruction_group_name:rotation sigma:0.3 equivariance_rate:30 orthogonality_rate:3 normalize:None normalized_l2:False l2_coef:1e-06/basis_layer:0',
                     './images/LEGACY/reconstruction_group_name:rotation sigma:0.5 equivariance_rate:30 orthogonality_rate:1 normalize:None normalized_l2:False l2_coef:1e-06/basis_layer:0',
                     './images/LEGACY/reconstruction_group_name:rotation sigma:0.5 equivariance_rate:30 orthogonality_rate:1 normalize:None normalized_l2:False l2_coef:1e-06/basis_layer:0',
                     './images/LEGACY/reconstruction_group_name:rotation sigma:0.3 equivariance_rate:10 orthogonality_rate:3 normalize:(None,\'Experiment1\') normalized_l2:False l2_coef:1e-06/basis_layer:0',
                     './images/LEGACY/reconstruction_group_name:rotation sigma:0.3 equivariance_rate:10 orthogonality_rate:1 normalize:(None,\'Experiment1\') normalized_l2:False l2_coef:1e-06/basis_layer:0')

    load_aux = [path_to_layer] * (len(basis_equiv_layers))
    if last_layer_type == 'conv1x1':
        onebyoneconv = [192, 192]
    elif last_layer_type == 'linear':
        basis_equiv_layers.extend([(0, 67, 1), (0, 67, 1)])
        stride_sz_conv.extend([1, 1])
        onebyoneconv = []
        # onebyoneconv = [192, 192]
    elif last_layer_type == 'group1x1':
        basis_equiv_layers.extend([(0, 67, 1), (0, 67, 1)])
        stride_sz_conv.extend([1, 1])
        onebyoneconv = []
    fc_sizes = []
else:
    raise NotImplementedError()

init_train(lr=lr, epochs=epochs, stride_sz_conv=stride_sz_conv,
           basis_equiv_layers=basis_equiv_layers,
           kernel_type='Gaussian', width=kernel_width, sigma=sigma,
           use_scipy_order2=False, save=save, save_aux=save_aux,
           equiv_rate=equiv_rate, orthg_rate=orthg_rate,
           weight_decay=weight_decay,
           normalize=normalize, verbose=verbose,
           train_basis_every_n_batches=train_basis_every_n_batches,
           train_basis_last_epoch=train_basis_last_epoch, bias=bias,
           group_name='rotation', nr_group_elems=8,
           fc_sizes=fc_sizes,
           normalized_l2=normalized_l2, dataset=dataset,
           target=target, load=load, load_aux=load_aux, onebyoneconv=onebyoneconv,
           basis_equiv_layers_type=basis_equiv_layers_type, pool_type=pool_type,
           last_layer_type=last_layer_type, finetune_batches=finetune_batches,
           rotation_augmentation_angle=rotation_augmentation_angle, optimizer=optimizer,
           rotation_augmentation_type=rotation_augmentation_type)
