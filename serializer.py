import pickle
import torch
import os

from basis_ae import BasisAE, Basis, init_basis_by_type
from basis_equivariant_layer import BasisEquivConvLyer
from basis_equivariant_network import BasisEquivariantNet
from constants import DEVICE

MODELS_FOLDER = './models/'


# TODO: change in order to save and load just from the parameters

class Serializer:
    def __init__(self):
        pass

    def save(self, model, config_dict, model_name, aux=None):
        path = os.path.join(MODELS_FOLDER, model_name)
        suffix = ''
        count = 0
        while os.path.exists(path + suffix):
            count += 1
            suffix = str(count)
        model_name = model_name + suffix
        path = path + suffix + '/'
        os.mkdir(path)

        # save model
        torch.save(model, path + 'model')
        # save parameters as objects
        with open(path + 'config_dict.pkl', 'wb+') as fd:
            pickle.dump(config_dict, fd)
        if aux:
            with open(path + 'aux.pkl', 'wb+') as fd:
                pickle.dump(aux, fd)
        # save parameters as text
        with open(path + 'config_dict.txt', 'w+') as fd:
            fd.write(str(config_dict))
        if aux:
            with open(path + 'aux.txt', 'w+') as fd:
                fd.write(str(aux))

        return model_name

    def load(self, model_name):
        path = os.path.join(MODELS_FOLDER, model_name)

        # load model
        model = torch.load(os.path.join(path, 'model'))

        # load other parameters
        with open(os.path.join(path, 'config_dict.pkl'), 'rb') as fd:
            config_dict = pickle.load(fd)
        if os.path.isfile(path + 'aux.pkl'):
            with open(os.path.join(path, 'aux.pkl'), 'rb') as fd:
                aux = pickle.load(fd)
        else:
            aux = None
        return model, config_dict, config_dict, aux

    def save_model_basis(self, model_name, model, config_dict):
        # TODO deal with frozen basis
        path = os.path.join('images', model_name)
        try:
            aux_model = model.module
        except AttributeError:
            aux_model = model

        for layer in aux_model.layers:
            if type(layer) == BasisEquivConvLyer and layer.filter_sz != 1:
                # Print model's state_dict
                print("Basis's state_dict:")
                # for param_tensor in layer.basis_ae.basis.state_dict():
                #     print(param_tensor, "\t", layer.basis_ae.basis.state_dict()[param_tensor].shape)
                torch.save(layer.basis_ae.basis.state_dict(), os.path.join(path, 'basis_layer:'+str(layer.basis_ae.index), 'basis.pt'))

                # save extra parameters as objects
                print("Basis's extra parameters:")
                with open(os.path.join(path, 'basis_layer:' + str(layer.basis_ae.index), 'extra_params.pkl'),
                          'wb+') as fd:
                    dict = {'FROZEN': layer.basis_ae.basis.FROZEN,
                            'normalize': layer.basis_ae.basis.normalize}
                    print(dict)
                    pickle.dump(dict, fd)

                # Print optimizer's state_dict
                print("BasisAE Optimizer's state_dict:")
                if layer.basis_ae.optimizer:
                    if layer.basis_ae.optimizer:
                        # for var_name in layer.basis_ae.optimizer.state_dict():
                        #     print(var_name)  # , "\t", layer.basis_ae.optimizer.state_dict()[var_name])
                        torch.save(layer.basis_ae.optimizer.state_dict(), os.path.join(path, 'basis_layer:'+str(layer.basis_ae.index), 'optimizer.pt'))

        self.save_config(model_name, config_dict)

    def load_model_basis(self, model, list_paths_to_basis, config_dict):
        assert config_dict.load == 'basis'
        config_dict.loaded_layers = []
        assert model.len_non1_basis_equiv_layers == model.len_non1_basis_equiv_layers

        for layer in model.layers:
            if type(layer) == BasisEquivConvLyer and layer.filter_sz != 1:
                path_to_layer = list_paths_to_basis[layer.basis_ae.index]
                if type(path_to_layer) not in [list, tuple]:
                    # Print model's state_dict
                    print("Basis's state_dict load:")
                    state_dict = torch.load(os.path.join(path_to_layer, 'basis.pt'))
                    if '_gamma' in state_dict.keys():
                        layer.basis_ae.basis.__setattr__('_gamma', torch.nn.Parameter(state_dict['_gamma'], True))
                    if '_beta' in state_dict.keys():
                        layer.basis_ae.basis.__setattr__('_beta', torch.nn.Parameter(state_dict['_beta'], True))

                    layer.basis_ae.basis.load_state_dict(state_dict)

                    # save extra parameters as objects
                    print("Basis's extra parameters load:")
                    with open(os.path.join(path_to_layer, 'extra_params.pkl'), 'rb+') as fd:
                        extra_parameters_state_dict = pickle.load(fd)
                        for key, val in extra_parameters_state_dict.items():
                            layer.basis_ae.basis.__setattr__(key, val)

                    # Print optimizer's state_dict
                    print("BasisAE Optimizer's state_dict load:")
                    # optimizer_state_dict = torch.load(os.path.join(path_to_layer, 'optimizer.pt'))
                    # for key, val in optimizer_state_dict.items():
                        # TODO set optimizer variables to cpu before saving!!
                        # layer.basis_ae.optimizer.__setattr__(key, val)
                    layer.basis_ae.optimizer = None

                    layer_idx_in_original_model = int(path_to_layer.split('/')[-1].split(':')[-1])
                    path_to_layers_net = '/'.join(path_to_layer.split('/')[:-1])
                    path_to_layers_config = os.path.join(path_to_layers_net, 'config_dict.pkl')
                    with open(path_to_layers_config, 'rb') as fd:
                        layer_config_dict = pickle.load(fd)

                    config_dict.loaded_layers.append({'layer_idx_in_original_model': layer_idx_in_original_model, 'config_dict_original_model': layer_config_dict})

                else:
                    #overcomplete basis
                    # nr basis = nr basis * size of a complete basis
                    square_basis_sz = layer.basis_ae.basis_sz * layer.basis_ae.basis_sz
                    assert layer.basis_ae.nr_basis == len(path_to_layer) * square_basis_sz
                    print("Loading overcomplete basis of "+str(len(path_to_layer)) + " basis")
                    for idx, actual_path_to_layer in enumerate(path_to_layer):
                        aux_basis, _ = init_basis_by_type('average', layer.basis_ae.basis_sz, square_basis_sz, layer.basis_ae.transformer.group.nr_group_elems, None)

                        # Print model's state_dict
                        print("Basis's state_dict load:")
                        state_dict = torch.load(os.path.join(actual_path_to_layer, 'basis.pt'))

                        if '_gamma' in state_dict.keys():
                            aux_basis.__setattr__('_gamma', torch.nn.Parameter(state_dict['_gamma'], True))
                        if '_beta' in state_dict.keys():
                            aux_basis.__setattr__('_beta', torch.nn.Parameter(state_dict['_beta'], True))

                        aux_basis.load_state_dict(state_dict)

                        print("Basis's extra parameters load:")
                        with open(os.path.join(actual_path_to_layer, 'extra_params.pkl'), 'rb+') as fd:
                            extra_parameters_state_dict = pickle.load(fd)
                            for key, val in extra_parameters_state_dict.items():
                                aux_basis.__setattr__(key, val)

                        aux_basis = aux_basis.to(DEVICE)
                        aux_basis.normalize_basis()
                        normalized_basis = aux_basis.get_normalized_basis()

                        layer.basis_ae.basis._unnormalized_basis[:, idx*square_basis_sz:(idx+1)*square_basis_sz] = normalized_basis

                    layer.basis_ae.basis.FROZEN = False
                    layer.basis_ae.basis._gamma = None
                    layer.basis_ae.basis._beta = None
                    layer.basis_ae.basis.normalize = None
                    layer.basis_ae.basis.freeze_basis()

    def save_config(self, model_name, config_dict, path=None):
        path = os.path.join('images', model_name) if path is None else path
        # save parameters as objects
        with open(path + '/config_dict.pkl', 'wb+') as fd:
            pickle.dump(config_dict, fd)
            # save parameters as text
        with open(path + '/config_dict.txt', 'w+') as fd:
            fd.write(str(config_dict))

    @staticmethod
    def load_config(path):
        path = path
        with open(path + '/config_dict.pkl', 'rb+') as fd:
            config_dict = pickle.load(fd)
        return config_dict

    def save_model(self, model_name, model, config_dict, aux=None, acc=None):
        path = os.path.join('images', model_name)
        try:
            aux_model = model.module
        except AttributeError:
            aux_model = model

        if aux is not None:
            path = os.path.join(path, aux)
            if not os.path.isdir(path):
                os.mkdir(path)
            path = os.path.join(path, 'acc={:.6f}'.format(acc))
            if not os.path.isdir(path):
                os.mkdir(path)
        self.save_config(model_name, config_dict, path)

        aux_model.freeze_basis()
        print("Model's state_dict saved:")
        # for param_tensor in aux_model.state_dict():
        #     print(param_tensor, "\t", aux_model.state_dict()[param_tensor].shape)
        torch.save(aux_model.state_dict(), os.path.join(path, 'model.pt'))

    def load_model(self, model: BasisEquivariantNet, path_to_model, config_dict):
        assert config_dict.load == 'model'
        state_dict = torch.load(os.path.join(path_to_model, 'model.pt'), map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)
        with open(os.path.join(path_to_model, 'config_dict.pkl'), 'rb') as fd:
            model_config_dict = pickle.load(fd)
        config_dict.loaded_model = model_config_dict
        for layer in model.layers:
            if type(layer) == BasisEquivConvLyer and layer.filter_sz != 1:
                layer.basis_ae.basis.FROZEN = True
