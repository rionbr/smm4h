__author__ = 'diegopinheiro'
__email__ = 'diegompin@gmail.com'
__github__ = 'https://github.com/diegompin'

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold


class KFoldCrossValidation(object):

    def __init__(self):
        self.n_splits = 10

    def get_cross_validation(self):
        cross_validation_kfold = StratifiedKFold(n_splits=self.n_splits)
        return cross_validation_kfold
