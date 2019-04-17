__author__ = 'diegopinheiro'
__email__ = 'diegompin@gmail.com'
__github__ = 'https://github.com/diegompin'

from src.algorithm_strategies.random_forest_strategy import RandomForestGridStrategy, RandomForestRandomizedStrategy
from src.training_strategies.training_strategy import CommonPipeline, PandasPipeline
from src.training_strategies.validation_strategies import KFoldCrossValidation
import pickle
import pandas as pd
import numpy as np


class DataLoadPickle(object):

    def __init__(self, filepath):
        self.filepath = filepath

    def get_data(self):
        with open(self.filepath, 'rb') as file:
            # file = open(self.filepath, 'rb')
            data = pickle.load(file)
            assert data is not None, 'data is none'
            data = data.astype(dtype=self.get_dtypes())
            col_target = 'y'
            data = data.loc[~data[col_target].isna()]
            y = data[col_target]
            y = y.astype(np.int)
            col_features = self.get_dtypes().keys()
            data = data.loc[:, col_features]
            X = data
            # data = data.fillna(0)
        return X, y

    def get_dtypes(self):
        dtypes = {
            "user_number_of_friends": np.float,
            # "user_log(number_of_friends)": np.float,
            "user_number_of_followers": np.float,  # np.int,
            # "user_log(number_of_followers)": np.float,
            "user_ratio_friends_followers": np.float,
            "user_number_of_tweets": np.float,  # np.int,
            # "user_log(number_of_tweets)": np.float,
            "temp_hour_of_day": 'category',
            "temp_day_of_week": 'category',
            "temp_season": 'category',
            "post_length_text": np.float,  # np.int,
            "post_number_words": np.float,  # np.int,
            "post_number_(NOUN+VERB+ADJ)": np.float,  # np.int,,
            "post_number_(Drugs)": np.float,  # np.int,
            "post_number_(MedicalTerms)": np.float,  # np.int,
            "post_number_(NaturalProducts)": np.float,  # np.int,
            "text_number_of_(VERB)": np.float,  # np.int,
            "text_number_of_(NOUN)": np.float,  # np.int,
            "text_number_of_(PRON)": np.float,  # np.int,
            "text_number_of_(ADJ)": np.float,  # np.int,
            "text_number_of_(ADV)": np.float,  # np.int,
            "text_number_of_(ADP)": np.float,  # np.int,
            "text_number_of_(CONJ)": np.float,  # np.int,
            "text_number_of_(DET)": np.float,  # np.int,
            "text_number_of_(NUM)": np.float,  # np.int,
            "text_number_of_(PRT)": np.float,  # np.int,
            "text_number_of_(X)": np.float,  # np.int,
            "text_number_of_(pct)": np.float,  # np.int,
            "post_tfidf(enbrel)": np.float,
            "post_tfidf(today)": np.float,
            "post_tfidf-parent(etanercept)": np.float,
            "post_tfidf-parent(water)": np.float
        }
        return dtypes

'''
filepath = 'data/task_1_train_features-110_sample.pickle'
self = DataLoadPickle(filepath)
'''


def main():
    filepath = 'data/task_1_train_features-110_sample.pickle'
    data_load = DataLoadPickle(filepath)
    X, y = data_load.get_data()
    cross_validation = KFoldCrossValidation()
    pipeline = PandasPipeline(X)
    learner = RandomForestGridStrategy(pipeline, cross_validation, n_jobs=8)
    models = learner.fit(X, y)
    print(models.cv_results_)


if __name__ == '__main__':
    # args = sys.argv[1]
    # main()
    pass