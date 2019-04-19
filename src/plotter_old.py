
import matplotlib.pyplot as plt

import numpy as np
import maracatu as m

model = 'randomforest'
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
sample = 'test' #train, test
scorer = 'f1_score'

path = 'datalink/task1'
path_results = f'{path}/results'
path_plots = f'{path}/plots'

model = 'random_forest_strategy'

import pickle
with open(f'{path_results}/{model}.pickle', 'rb') as file:
    results = pickle.load(file)

values = get_values(results, scorer, sample, kfold)

plot_hist(values, 'F1_score (X)', 'P(X)', path_plots, f'{model}_{sample}_{scorer}.pdf')
