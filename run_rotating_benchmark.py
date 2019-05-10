import math
import os
import pickle
import sys

from serializer import Serializer
from trainer import init_trainer_model


def get_models_list(path_to_models):
    models_to_process = []
    all_models = os.listdir(path_to_models)
    if 'LEGACY' in all_models:
        all_models.remove('LEGACY')
    for model in all_models:
        path_to_model = os.path.join(path_to_models, model)
        if not os.path.isdir(path_to_model):
            continue

        path_to_best = os.path.join(path_to_model, 'best')
        best_list = os.listdir(path_to_best)
        # remove dirs with acc less than 10
        for i in range(10):
            best_list = [el for el in best_list if '=' + str(i) + '.' not in el]

        best_list.sort()
        best_model = best_list[-1]
        best_model_path = os.path.join(path_to_best, best_model)

        models_to_process.append(best_model_path)
    return models_to_process


def load_model(model_path):
    config_dict = Serializer.load_config(model_path)
    config_dict.load = 'model'
    config_dict.load_aux = model_path
    config_dict.save = None
    config_dict.save_aux = None
    config_dict.epochs = 1
    if 'optimizer' not in config_dict.__dict__.keys():
        config_dict.optimizer = 'adam'
    if 'finetune_batches' not in config_dict.__dict__.keys():
        config_dict.finetune_batches = 0
    if 'last_layer_type' not in config_dict.__dict__.keys():
        config_dict.last_layer_type = 'conv1x1'
    if 'pool_type' not in config_dict.__dict__.keys():
        config_dict.pool_type = 'stride'
    if 'basis_equiv_layers_type' not in config_dict.__dict__.keys():
        config_dict.basis_equiv_layers_type = 'conv'
    if 'dataset' not in config_dict.__dict__.keys():
        config_dict.dataset = 'CIFAR10'
    if 'rotation_augmentation_angle' not in config_dict.__dict__.keys():
        config_dict.rotation_augmentation_angle = '0'
    if 'rotation_augmentation_type' not in config_dict.__dict__.keys():
        config_dict.rotation_augmentation_type = 'torch'
    if 'equivariance_rate' in config_dict.__dict__.keys():
        config_dict.equiv_rate = config_dict.equivariance_rate
        del config_dict.equivariance_rate
    if 'orthogonality_rate' in config_dict.__dict__.keys():
        config_dict.orthg_rate = config_dict.orthogonality_rate
        del config_dict.orthogonality_rate
    for key in ['shape_input', 'loaded_layers']:
        if key in config_dict.__dict__.keys():
            del config_dict.__dict__[key]
    config_dict.train_basis_last_epoch = -2  # 0 if tou want to show eq, rec loss
    verbose = False

    trainer, model, transformer, config_dict = init_trainer_model(**config_dict.__dict__, verbose=verbose,
                                                                  target='prediction')
    return trainer, model


# sys.argv[1] main folder
# sys.argv[2] dataset
# sys.argv[3] augmentation
# TODO change this before cluster
root_folder_results = os.path.join('./images', sys.argv[1])

path_to_models = root_folder_results

# root_folder_results = os.path.join('./Downloaded models', sys.argv[1])
# if sys.argv[3] == '0':
#     root_folder_results = os.path.join(root_folder_results, '0')
# else:

# root_folder_results = os.path.join(root_folder_results, sys.argv[3])
#
# if sys.argv[1] == 'augmentation and finetuned/augmentation_type gaussian':
#     if len(sys.argv) == 5 and sys.argv[4] == 'conv_only':
#         path_to_models = os.path.join(root_folder_results, '0_batches', 'conv_only', sys.argv[2])
#     else:
#         path_to_models = os.path.join(root_folder_results, '0_batches', sys.argv[2])
# elif sys.argv[1] == 'benchmarks':
#     path_to_models = os.path.join(root_folder_results, sys.argv[2])
# else:
#     raise NotImplementedError


# results_dict_path = os.path.join(path_to_models, 'results_dict1.pkl')

models_to_process = get_models_list(path_to_models)

results_dict = {}
for idx, model_path in enumerate(models_to_process):
    trainer, model = load_model(model_path)
    if trainer.transformer.rotation_batch_tansformer.interpolator.kernel.sigma != 0.35:
        trainer.transformer.rotation_batch_tansformer.interpolator.kernel.sigma = 0.35
    results_dict[model_path] = []
    for angle in range(0, 360, 1):
        _, acc = trainer.validate(model, -1, angle=float(2 * math.pi / 360 * angle))
        results_dict[model_path].append(acc)

    results_dict_path = os.path.join(path_to_models, 'results_dict'+str(idx)+'.pkl')
    with open(results_dict_path, 'wb+') as fd:
        pickle.dump(results_dict, fd)

# with open(results_dict_path, 'wb+') as fd:
#     pickle.dump(results_dict, fd)
