__author__ = 'diegopinheiro'
__email__ = 'diegompin@gmail.com'
__github__ = 'https://github.com/diegompin'

from src.training_strategies.search_strategy import GridSearchStrategy, RandomizedSearchStrategy
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from scipy.stats import randint as sp_randint


class RandomForestGridStrategy(GridSearchStrategy):

    def __init__(self, pipeline, cross_validation, n_jobs=None):
        super().__init__(pipeline, cross_validation, n_jobs)

    def get_classifier(self):
        return RandomForestClassifier()

    # TODO Define the parameter space
    def get_params(self):
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        params = {
            'classifier__n_estimators': n_estimators,
            'classifier__max_features': max_features,
            'classifier__max_depth': max_depth,
            'classifier__min_samples_split': min_samples_split,
            'classifier__min_samples_leaf': min_samples_leaf,
            'classifier__bootstrap': bootstrap
        }

        return params


class RandomForestRandomizedStrategy(RandomizedSearchStrategy):

    def __init__(self, pipeline, cross_validation, n_jobs):
        super().__init__(pipeline, cross_validation, n_jobs)

    # TODO Define the parameter space
    def get_params(self):
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        params = {
            'classifier__n_estimators': n_estimators,
            'classifier__max_features': max_features,
            'classifier__max_depth': max_depth,
            'classifier__min_samples_split': min_samples_split,
            'classifier__min_samples_leaf': min_samples_leaf,
            'classifier__bootstrap': bootstrap
        }

        return params

    def get_classifier(self):
        return RandomForestClassifier()
