__author__ = 'diegopinheiro'
__email__ = 'diegompin@gmail.com'
__github__ = 'https://github.com/diegompin'

from abc import ABCMeta
from abc import abstractmethod
import six
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing.imputation import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


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


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)


class TypeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])


class PandasPipeline(CommonPipeline):

    def __init__(self, cols_feature):
        super().__init__()
        self.cols_feature = cols_feature

    def get_pipeline(self, classifier):
        # preprocess_pipeline = make_pipeline(
        #     ColumnSelector(columns=self.cols_feature),
        #     ,
        # )

        feature_union = FeatureUnion(transformer_list=[
            ("numeric_features", make_pipeline(
                TypeSelector(np.number),
                Imputer(strategy="median"),
                StandardScaler()
            )),
            ("categorical_features", make_pipeline(
                TypeSelector("category"),
                Imputer(strategy="most_frequent"),
                OneHotEncoder()
            )),
            ("boolean_features", make_pipeline(
                TypeSelector("bool"),
                Imputer(strategy="most_frequent")
            ))
        ])

        pipeline = Pipeline(steps=[
            ('colselector', ColumnSelector(columns=self.cols_feature)),
            ('featureunion', feature_union),
            ('classifier', classifier)
        ])

        # make_pipeline(
        #     preprocess_pipeline,
        #     'classfier': classifier,
        # )

        return pipeline
