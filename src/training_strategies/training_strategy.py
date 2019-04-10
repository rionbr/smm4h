__author__ = 'diegopinheiro'
__email__ = 'diegompin@gmail.com'
__github__ = 'https://github.com/diegompin'





from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

from abc import ABCMeta
from abc import abstractmethod
import six


class TrainingStrategy(six.with_metaclass(ABCMeta)):

    def __init__(self, pipeline, cross_validation):
        self.pipeline = pipeline
        self.cross_validation = cross_validation

    @abstractmethod
    def get_classifier(self):
        pass

    def get_pipeline(self):
        classifier = self.get_classifier()
        return self.pipeline.get_pipeline(classifier)

    def get_cross_validation(self):
        return self.cross_validation.get_cross_validation()

    @abstractmethod
    def get_search(self):
        pass

    # @abstractmethod
    # def get_cross_validation(self):
    #     pass

    def fit(self, X, y):
        search = self.get_search()
        return search.fit(X, y)


class PipelineStrategy(six.with_metaclass(ABCMeta)):

    def __init__(self):
        pass

    @abstractmethod
    def get_pipeline(self, classifier):
        pass


class CommonPipeline(PipelineStrategy):

    def __init__(self):
        super().__init__()

    def get_pipeline(self, classifier):
        estimators = list()
        estimators.append(('classifier', classifier))
        model_pipeline = Pipeline(estimators)
        return model_pipeline


class TextPipeline(PipelineStrategy):

    def __init__(self):
        super().__init__()

    def get_pipeline(self, classifier):
        estimators = list()
        estimators.append(('vect', CountVectorizer()))
        estimators.append(('tfidf', TfidfTransformer()))
        estimators.append(('classifier', classifier))
        model_pipeline = Pipeline(estimators)
        return model_pipeline
