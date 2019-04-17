__author__ = 'diegopinheiro'
__email__ = 'diegompin@gmail.com'
__github__ = 'https://github.com/diegompin'

from abc import ABCMeta
from abc import abstractmethod
import six

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from src.training_strategies.training_strategy import TrainingStrategy


class GridSearchStrategy(six.with_metaclass(ABCMeta, TrainingStrategy)):

    def __init__(self, pipeline, cross_validation):
        super().__init__(pipeline, cross_validation)

    @abstractmethod
    def get_params(self):
        pass

    def get_search(self):
        cross_validation = self.get_cross_validation()
        pipeline = self.get_pipeline()
        params = self.get_params()
        search = GridSearchCV(estimator=pipeline, cv=cross_validation, param_grid=params)
        return search


class RandomizedSearchStrategy(six.with_metaclass(ABCMeta, TrainingStrategy)):

    def __init__(self, pipeline, cross_validation):
        super().__init__(pipeline, cross_validation)

    @abstractmethod
    def get_params(self):
        pass

    def get_search(self):
        cross_validation = self.get_cross_validation()
        pipeline = self.get_pipeline()
        params = self.get_params()
        search = RandomizedSearchCV(estimator=pipeline, cv=cross_validation, param_distributions=params)
        return search
