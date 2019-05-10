import math
import pickle

from trainer import TrainerPrediction, init_train, init_trainer_model
from serializer import Serializer

# model_path = './images/finetuned/new/0_batches/CIFAR10/prediction_dataset:CIFAR10_conv_type:learned_pool_type:max last_layer_type:linear sigma:0.5 basis_sz:[(9,33,3),(9,33,3),(9,33,3),(9,67,3),(9,67,3),(9,67,3),(9,67,3),(0,67,1),(0,67,1)] equivariance_rate:None orthogonality_rate:None load:basis/best/acc=92.750000'
# model_path = './images/finetuned/new/0_batches/CIFAR10/prediction_dataset:CIFAR10_conv_type:average_pool_type:max last_layer_type:linear finetune_batches:0 load:basis/best/acc=90.930000'
model_path = './images/benchmarks/new/CIFAR10/prediction_dataset:CIFAR10_conv_type:bilinear_pool_type:max last_layer_type:linear finetune_batches:0 load:basis/best/acc=90.630000'
# model_path = './Downloaded models/ALL BASIS TRAIN/striving for simplicity/prediction_group_name:rotation sigma:0.5 equivariance_rate:10 orthogonality_rate:1 normalize:None normalized_l2:None l2_coef:1e-06 load:None/best/acc=91.750000'
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
if 'equivariance_rate' in config_dict.__dict__.keys():
    config_dict.equiv_rate = config_dict.equivariance_rate
    del config_dict.equivariance_rate
if 'orthogonality_rate' in config_dict.__dict__.keys():
    config_dict.orthg_rate = config_dict.orthogonality_rate
    del config_dict.orthogonality_rate
if 'rotation_augmentation_angle' not in config_dict.__dict__.keys():
    config_dict.rotation_augmentation_angle = '0'
if 'rotation_augmentation_type' not in config_dict.__dict__.keys():
    config_dict.rotation_augmentation_type = 'torch'

for key in ['shape_input', 'loaded_layers']:
    if key in config_dict.__dict__.keys():
        del config_dict.__dict__[key]

config_dict.train_basis_last_epoch = -2 # 0 if tou want to show eq, rec loss
verbose = True

config_dict.finetune_batches = 0
trainer, model, transformer, config_dict = init_trainer_model(**config_dict.__dict__, verbose=verbose, target='prediction')
trainer.train(model=model, transformer=transformer, config_dict=config_dict)
# trainer.finetune_basis(model)
trainer.validate(model, -1, angle=None, eq_loss=True)

