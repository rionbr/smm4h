__author__ = 'diegopinheiro'
__email__ = 'diegompin@gmail.com'
__github__ = 'https://github.com/diegompin'


from src.training_strategies.search_strategy import GridSearchStrategy, RandomizedSearchStrategy
from sklearn.ensemble import RandomForestClassifier

from scipy.stats import randint as sp_randint


class RandomForestGridStrategy(GridSearchStrategy):

    def __init__(self, pipeline, cross_validation):
        super().__init__(pipeline, cross_validation)

    def get_classifier(self):
        return RandomForestClassifier()

    #TODO Define the parameter space
    def get_params(self):
        params = {
            'classifier__n_estimators': [1, 10],
            'classifier__max_leaf_nodes': [5, 10]
        }
        return params


class RandomForestRandomizedStrategy(RandomizedSearchStrategy):

    def __init__(self, pipeline, cross_validation):
        super().__init__(pipeline, cross_validation)

    # TODO Define the parameter space
    def get_params(self):
        params = {
            'classifier__n_estimators': sp_randint(1, 10),
            'classifier__max_leaf_nodes': sp_randint(5, 10)
        }
        return params

    def get_classifier(self):
        return RandomForestClassifier()
