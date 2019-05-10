import csv
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('folder', type=str,
                    help='a folder name of the models. Each layer subfolder containing an output.txt')
parser.add_argument('nr_layers', type=int,
                    help='the number of layers of each model in the folder')
args = parser.parse_args()

path = os.path.join('./', args.folder)
os.chdir(path)

model_names = os.listdir('./')
#remove hidden folders:
model_names = [name for name in model_names if not name.startswith('.')]
idx = 0
for layer_idx in range(args.nr_layers):
    layer_results_path = 'layer:' + str(layer_idx) + ' results.csv'
    with open(layer_results_path, 'w') as csvfile:
        print('created results file')

    for model_name in model_names:
        if not os.path.isdir(model_name):
            continue
        idx += 1
        model_layer_path = os.path.join(model_name, 'basis_layer:'+str(layer_idx), 'output.txt')
        with open(model_layer_path, 'r') as fd:
            model_stats = fd.read()

        model_stats = model_stats.split('\n')
        while '' in model_stats:
            model_stats.remove('')
        while ' ' in model_stats:
            model_stats.remove(' ')
        if len(model_stats) == 0:
            continue

        model_description = model_name
        # model_description = model_description.replace(' ', '_', 1)
        # model_description = model_description.replace(', ', ',')
        # model_description = model_description.replace('Experiment ', 'Experiment', 1)
        model_description = model_description.split(' ')
        for i in range(len(model_description)):
            model_description[i] = model_description[i].split(':')

        losses = [model_stats[i] for i in range(int(len(model_stats)))]

        for i in range(len(losses)):
            losses[i] = losses[i].split('\t')
            for j in range(len(losses[i])):
                if j == 0:
                    continue
                losses[i][j] = losses[i][j].split(':')

        if idx == 1:
            with open(layer_results_path, 'a+') as csvfile:
                spamwriter = csv.writer(csvfile)

                row = ['model_name']
                row.extend([model_description[i][0] for i in range(len(model_description))])
                [row.extend([losses[i][0]] * 6) for i in reversed(range(len(losses)))]
                spamwriter.writerow(row)

            with open(layer_results_path, 'a+') as csvfile:
                spamwriter = csv.writer(csvfile)

                row = [' ']
                row.extend([' ' for i in range(len(model_description))])
                row.extend([losses[i][j + 1][0] for i in reversed(range(len(losses))) for j in range(len(losses[i]) - 1)])
                spamwriter.writerow(row)

        with open(layer_results_path, 'a+') as csvfile:
            spamwriter = csv.writer(csvfile)

            row = [model_name]
            row.extend([model_description[i][1] for i in range(len(model_description))])
            row.extend([losses[i][j + 1][1] for i in reversed(range(len(losses))) for j in range(len(losses[i]) - 1)])
            spamwriter.writerow(row)

        pass

