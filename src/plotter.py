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
filepath = 'results/task1/random_forest_strategy.pickle'
with open(filepath, 'rb') as file:
    # file = open(self.filepath, 'rb')
    data = pickle.load(file)