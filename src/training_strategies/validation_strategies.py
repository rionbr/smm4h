__author__ = 'diegopinheiro'
__email__ = 'diegompin@gmail.com'
__github__ = 'https://github.com/diegompin'

from sklearn.model_selection import KFold


class KFoldCrossValidation(object):

    def __init__(self, n_splits):
        self.n_splits = n_splits

    def get_cross_validation(self):
        cross_validation_kfold = KFold(n_splits=self.n_splits)
        return cross_validation_kfold
