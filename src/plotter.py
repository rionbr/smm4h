__author__ = 'diegopinheiro'
__email__ = 'diegompin@gmail.com'
__github__ = 'https://github.com/diegompin'


import matplotlib.pyplot as plt
import numpy as np

class Plotter(object):

    def __init__(self):
        pass


    # TODO
    def plot(self, scores):
        from statsmodels.distributions.empirical_distribution import ECDF
        fig, ax = plt.subplots()
        ecdf = ECDF(scores)
        bins = np.linspace(0, 1)
        ax.plot(bins, ecdf(bins))
        plt.show()

    def bootstrap_distribution(self, scores):
        x = scores
        n = len(scores)
        reps = 1000
        xb = np.random.choice(x, (n, reps))
        mb = xb.mean(axis=0)
        mb.sort()
        p_25, p_975 = np.percentile(mb, [2.5, 97.5])
        return mb, p_25, p_975

    def plot_bootstrap(self, mb):
        fig, ax = plt.subplots()
        ax.hist(mb)
        plt.show()


import pickle
filepath = 'datalink/task1/random_forest_strategy.pickle'
with open(filepath, 'rb') as file:
    # file = open(self.filepath, 'rb')
    data = pickle.load(file)
    results = data.cv_results_


import matplotlib.pyplot as plt
fig
model = 'Random Forest '
plt.figure(figsize=(13, 13))
plt.title(f"{model}", fontsize=16)

plt.xlabel("min_samples_split")
plt.ylabel("Score")

import maracatu as m


def get_par_base():
    par_base = {
        'xvar': 'X',
        'yvar': 'Y',
        'title': 'Titanium Patients',
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

    plt.show()

scorer = 'f1_score'
for sample, style in (('train', '--'), ('test', '-')):
    sample_score_mean = results['mean_%s_%s' % (sample, scorer)]

