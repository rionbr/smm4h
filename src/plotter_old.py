
import matplotlib.pyplot as plt

import numpy as np
import maracatu as m
import pickle
import itertools as it
# plt.figure(figsize=(13, 13))
# plt.title(f"{model}", fontsize=16)

def get_par_base():
    par_base = {
        'xvar': 'X',
        'yvar': 'Y',
        'title': '',
        'xlabel': 'Date',

        'alpha': 1,
        'grid': True,
        'fig.tight_layout': True,
        'fig.figsize': (20, 10),
    }
    return par_base

def plot_hist(mb, xlabel, ylabel, filepath, filename):
    par_base = get_par_base()
    par_base.update({
        'file_output': f'{filepath}/{filename}',
        'xlabel': f'{xlabel}',
        'ylabel': f'{ylabel}'
    })

    fig, ax = plt.subplots(1, 1)

    ax.hist(mb)

    plt_base = m.PlotBase()
    plt_base.configure_ax(ax, par_base)
    plt_base.configure_fig(fig, par_base)


    plt.savefig(filename)

def get_values(results, scorer, sample, kfold):
    values = [results.cv_results_[f'split{i}_{sample}_{scorer}'] for i in range(0, kfold)]
    return np.ravel(values)

kfold = 10

path = 'datalink/'
path_results = f'{path}/results'
path_plots = f'{path}/plots'

task = 'task1'
data_set = 'train'
sample = 'test' #train, test
scorer = ['f1_score', 'precision', 'recall', 'roc_auc_score', 'matthews_corrcoef'] #

model = 'random_forest_strategy'

with open(f'{path_results}/{model}.pickle', 'rb') as file:
    results = pickle.load(file)



for s in scorer:
    values = get_values(results, s, sample, kfold)
    plot_hist(values, f'{s} (X)', 'P(X)', path_plots, f'{model}_{sample}_{s}.pdf')
