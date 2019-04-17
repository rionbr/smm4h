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
            data = data.astype(dtype=self.get_dtypes())
            data = data[self.get_dtypes().keys()]
            assert data is not None, 'data is none'
            col_target = 'y'
            col_features = data.columns != col_target
            data = data.loc[~data[col_target].isna()]
            # data = data.fillna(0)
            y = data[col_target]
            X = data.loc[:, col_features]
        return X, y

    def get_dtypes(self):
        dtypes = {
            'y': np.float,
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

    np.isfinite(X.select_dtypes(include=[np.float]))



    # from sklearn.ensemble.forest import RandomForestClassifier
    # clf = RandomForestClassifier()
    # clf.fit(X, y)
    cross_validation = KFoldCrossValidation(n_splits=10)
    pipeline = PandasPipeline(X.columns)

    learner = RandomForestGridStrategy(pipeline, cross_validation)


    print(learner.fit(X, y))


if __name__ == '__main__':
    # data = sys.argv[1]
    # pass
    main()

'''
import pandas as pd
import numpy as np
desired_width = 620
pd.set_option('display.width', desired_width)
pd.set_option("display.max_columns", 10)
np.set_printoptions(linewidth=desired_width)

'''

X['temp_hour_of_day'].dtype
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False, n_values=20)
a = encoder.fit_transform(X[['temp_season']])
a