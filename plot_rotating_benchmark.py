import os
import sys

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pickle

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# sns.set()

plot = 'augmentation and finetuned/augmentation_type gaussian' # 'benchmark'

if plot == 'benchmark':
    augmentation_angle = '0'
    fig, (ax1, ax2) = plt.subplots(2, 1, sharey=False)
    axs = [ax1, ax2]
    for idx, dataset in enumerate(['MNIST', 'CIFAR10']):
        # fig, ax = plt.subplots()
        ax = axs[idx]
        ax.set_yscale('log')
        if dataset == 'MNIST':
            ax.set_ylim(0.3, 100)
        elif dataset == 'CIFAR10':
            ax.set_ylim(9.9, 100)
        ax.set_xlim(0, 360)
        plt.xticks([45 * i for i in range(int(360 / 45)+1)])
        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        plt.gca().get_yaxis().set_minor_formatter(plt.NullFormatter())
        ax.set_ylabel('Validation Set Error (%)')
        ax.set_xlabel('Validation Set Angle')
        for finetuned_batches in ['benchmarks', 'augmentation and finetuned/augmentation_type torch']:  # , '400', '100', 'not_finetuned']:  #]:  # , 'not_finetuned_new']:
            # if finetuned_batches not in ['not_finetuned', 'benchmarks']:
            #     with open('./Downloaded models/finetuned/new/' + str(
            #             finetuned_batches) + '_batches/' + dataset + ' finetune_batches:' + finetuned_batches + ' results_dict.pkl',
            #               'rb') as fd:
            #         results_dict = pickle.load(fd)
            # elif finetuned_batches == 'not_finetuned':
            #     with open('./Downloaded models/not_finetuned/results_dict.pkl', 'rb') as fd:
            #         results_dict = pickle.load(fd)
            #
            #     our_model = './images/prediction_dataset:MNIST_conv_type:learned_pool_type:max last_layer_type:linear sigma:0.5 basis_sz:[(9,33,3),(9,33,3),(9,33,3),(9,67,3),(9,67,3),(9,67,3),(9,67,3),(0,67,1),(0,67,1)] equivariance_rate:None orthogonality_rate:None load:basis/best/acc=99.630000'
            #
            #     for key in list(results_dict.keys()):
            #         if key != our_model:
            #             results_dict.pop(key)
            # elif finetuned_batches == 'benchmarks':
            #     with open('./images/benchmarks/new/' + dataset + '/results_dict.pkl', 'rb') as fd:
            #         results_dict = pickle.load(fd)
            # else:
            #     pass

            root_folder_results = os.path.join('./images', finetuned_batches)
            # root_folder_results = os.path.join('./Downloaded models', sys.argv[1])
            if augmentation_angle == '0':
                root_folder_results = os.path.join(root_folder_results, 'new')
            else:
                root_folder_results = os.path.join(root_folder_results, augmentation_angle)

            if 'augmentation and finetuned' in finetuned_batches:
                path_to_models = os.path.join(root_folder_results, '0_batches', dataset)
            elif finetuned_batches == 'benchmarks':
                path_to_models = os.path.join(root_folder_results, dataset)
            else:
                raise NotImplementedError

            results_dict_path = os.path.join(path_to_models, 'results_dict.pkl')
            with open(results_dict_path, 'rb') as fd:
                results_dict = pickle.load(fd)

            # plt.figure()
            for idx, model in enumerate(results_dict.keys()):
                if 'pool_type:avg' in model or 'pool_type:stride' in model:
                    continue
                print(idx, model)

                str_model = ''
                # str_model = str(idx) + ' '
                # if '400_batches' in model:
                #     str_model += '400 '
                # elif '100_batches' in model:
                #     str_model += '100 '
                # elif '0_batches' in model:
                #     str_model += '0 '
                if 'conv_type:average' in model:
                    str_model += '$\\bf{Partial (ours)}$'
                    line_style = '-'
                if 'conv_type:random' in model:
                    str_model += 'Random '
                    line_style = '--'
                if 'conv_type:learned' in model:
                    str_model += '$\\bf{Full (ours)}$'
                    line_style = '-'
                if 'conv_type:weiler' in model:
                    str_model += 'Weiler '
                    line_style = '-.'
                if 'conv_type:conv' in model:
                    str_model += 'Conv '
                    line_style = '--'
                if 'conv_type:bilinear' in model:
                    str_model += 'Bilinear '
                    line_style = '-.'
                if 'conv_type:gaussian' in model:
                    str_model += 'Gaussian '
                    line_style = '-.'
                # if 'pool_type:avg' in model:
                #     str_model += 'avg '
                # if 'pool_type:max' in model:
                #     str_model += 'max '
                # if 'last_layer_type:group1x1' in model:
                #     str_model += 'group1x1 '
                # if 'last_layer_type:conv1x1' in model:
                #     str_model += 'conv1x1 '
                # if 'last_layer_type:linear' in model:
                #     str_model += 'linear '

                ax.plot([i for i in range(0, 360, 1)], [100 - acc for acc in results_dict[model]], label=str_model, ls=line_style)

            # plt.gca().invert_yaxis()
            # plt.legend(loc='upper right')
            # plt.title(dataset + ' finetuned ' + str(finetuned_batches))
            # plt.tight_layout()
            pass

    ax1.set_xticks([])
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax1.set_ylabel(ax1.get_ylabel(), fontsize=12)
    ax2.set_ylabel(ax2.get_ylabel(), fontsize=12)
    ax2.set_xlabel(ax2.get_xlabel(), fontsize=12)

    all_handles, all_labels = ax.get_legend_handles_labels()
    handles = all_handles[0:4]
    labels = all_labels[0:4]

    legend1 = plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.3),
                         fancybox=True, shadow=True, ncol=5)

    handles = all_handles[4:]
    labels = all_labels[4:]

    legend2 = plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.5),
                         fancybox=True, shadow=True, ncol=5)
    ax2.add_artist(legend1)
    ax2.add_artist(legend2)
    plt.tight_layout()
    pass

elif plot == 'data_augmentation' or plot == 'augmentation and finetuned/augmentation_type gaussian':
    fig, (ax1, ax2) = plt.subplots(2, 1, sharey=False)
    axs = [ax1, ax2]
    for idx, dataset in enumerate(['MNIST', 'CIFAR10']):
        # fig, ax = plt.subplots()
        ax = axs[idx]
        ax.set_yscale('log')
        if dataset == 'MNIST':
            ax.set_ylim(0.4, 10)
        elif dataset == 'CIFAR10':
            ax.set_ylim(9.9, 100)
        ax.set_xlim(0, 360)
        plt.xticks([45 * i for i in range(int(360 / 45) + 1)])
        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        plt.gca().get_yaxis().set_minor_formatter(plt.NullFormatter())
        ax.set_ylabel('Validation Set Error (%)')
        ax.set_xlabel('Validation Set Angle')

        for angle in ['0', '45', '90', 'all']:  # , '400', '100', 'not_finetuned']:  #]:  # , 'not_finetuned_new']:
            # if finetuned_batches not in ['not_finetuned', 'benchmarks']:
            #     with open('./Downloaded models/finetuned/new/' + str(
            #             finetuned_batches) + '_batches/' + dataset + ' finetune_batches:' + finetuned_batches + ' results_dict.pkl',
            #               'rb') as fd:
            #         results_dict = pickle.load(fd)
            # elif finetuned_batches == 'not_finetuned':
            #     with open('./Downloaded models/not_finetuned/results_dict.pkl', 'rb') as fd:
            #         results_dict = pickle.load(fd)
            #
            #     our_model = './images/prediction_dataset:MNIST_conv_type:learned_pool_type:max last_layer_type:linear sigma:0.5 basis_sz:[(9,33,3),(9,33,3),(9,33,3),(9,67,3),(9,67,3),(9,67,3),(9,67,3),(0,67,1),(0,67,1)] equivariance_rate:None orthogonality_rate:None load:basis/best/acc=99.630000'
            #
            #     for key in list(results_dict.keys()):
            #         if key != our_model:
            #             results_dict.pop(key)
            # elif finetuned_batches == 'benchmarks':
            #     with open('./images/benchmarks/new/' + dataset + '/results_dict.pkl', 'rb') as fd:
            #         results_dict = pickle.load(fd)
            # else:
            #     pass

            # root_folder_results = os.path.join('./images', 'finetuned')
            root_folder_results = os.path.join('./images/augmentation and finetuned/augmentation_type gaussian')
            if angle == '0':
                # root_folder_results = os.path.join(root_folder_results, 'new')
                root_folder_results = os.path.join(root_folder_results, '0')
            else:
                root_folder_results = os.path.join(root_folder_results, angle)

            path_to_models = os.path.join(root_folder_results, '0_batches', dataset)

            results_dict_path = os.path.join(path_to_models, 'results_dict.pkl')
            with open(results_dict_path, 'rb') as fd:
                results_dict = pickle.load(fd)

            # plt.figure()
            for idx, model in enumerate(results_dict.keys()):
                line_style = '-'
                if 'pool_type:avg' in model or 'pool_type:stride' in model:
                    continue
                if 'conv_type:learned' in model:
                    continue
                print(idx, model)

                str_model = ''

                # str_model = str(idx) + ' '
                # if '400_batches' in model:
                #     str_model += '400 '
                # elif '100_batches' in model:
                #     str_model += '100 '
                # elif '0_batches' in model:
                #     str_model += '0 '

                # if 'conv_type:average' in model:
                #     # str_model += '$\\bf{average (ours)}$'
                #     line_style = '-'
                # if 'conv_type:random' in model:
                #     str_model += 'random '
                #     line_style = '--'
                # if 'conv_type:learned' in model:
                #     str_model += '$\\bf{learned (ours)}$'
                #     line_style = '-'

                if angle == '0':
                    augm_str = ' no aug'
                elif angle == 'all':
                    augm_str = ' full aug'
                else:
                    augm_str = angle
                str_model += augm_str
                # if 'pool_type:avg' in model:
                #     str_model += 'avg '
                # if 'pool_type:max' in model:
                #     str_model += 'max '
                # if 'last_layer_type:group1x1' in model:
                #     str_model += 'group1x1 '
                # if 'last_layer_type:conv1x1' in model:
                #     str_model += 'conv1x1 '
                # if 'last_layer_type:linear' in model:
                #     str_model += 'linear '

                ax.plot([i for i in range(0, 360, 1)], [100 - acc for acc in results_dict[model]],
                        label=str_model, ls=line_style)
                print(str_model)

            # plt.gca().invert_yaxis()

            # plt.title(dataset + ' finetuned ' + str(finetuned_batches))
            plt.tight_layout()
            pass

    ax1.set_xticks([])
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax1.set_ylabel(ax1.get_ylabel(), fontsize=12)
    ax2.set_ylabel(ax2.get_ylabel(), fontsize=12)
    ax2.set_xlabel(ax2.get_xlabel(), fontsize=12)

    handles, labels = ax.get_legend_handles_labels()
    handles = [handles[0], handles[2], handles[1], handles[3]]
    labels = [labels[0], labels[2], labels[1], labels[3]]

    # plt.legend(handles, labels, loc=2)
    # plt.legend(loc='upper right')

    plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.25),
               fancybox=True, shadow=True, ncol=5)
    plt.tight_layout()

    pass