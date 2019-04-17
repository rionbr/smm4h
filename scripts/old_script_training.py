__author__ = 'diegopinheiro'
__email__ = 'diegompin@gmail.com'
__github__ = 'https://github.com/diegompin'


from src.algorithm_strategies.random_forest_strategy import RandomForestRandomizedStrategy
from src.training_strategies.training_strategy import TextPipeline
from src.training_strategies.validation_strategies import KFoldCrossValidation
import pickle
import pandas as pd


class DataLoadPickle(object):

    def __init__(self, filepath):
        self.filepath = filepath

    def get_data(self):
        with open(self.filepath, 'rb') as file:
            # file = open(filepath, 'rb')
            data = pickle.load(file)
            assert data is not None, 'data is none'
            col_target = 'y'
            col_features = data.columns != col_target
            data = data.loc[~data[col_target].isna()]
            data = data.fillna(0)
            y = data[col_target].values
            X = data.loc[:, col_features].values
        return X, y

'''
filepath = 'data/task_1_train_features-110_sample.pickle'
self = DataLoadPickle(filepath)
'''

def main():
    filepath = 'data/task_1_train_features-110_sample.pickle'
    data_load = DataLoadPickle(filepath)
    X, y = data_load.get_data()


    # TODO Change for tweets
    def get_data():
        from sklearn.datasets import fetch_20newsgroups
        cats = ['alt.atheism', 'sci.space']
        newsgroups_train = fetch_20newsgroups(subset='train', categories=cats)
        newsgroups_test = fetch_20newsgroups(subset='test', categories=cats)
        X_train = newsgroups_train.data
        X_test = newsgroups_test.data
        y_train = newsgroups_train.target
        y_test = newsgroups_test.target
        return X_train[0:30], y_train[0:30]

    # def get_data():
    #     pass

    X, y = get_data()

    cross_validation = KFoldCrossValidation(n_splits=10)
    pipeline = TextPipeline()
    learner = RandomForestRandomizedStrategy(pipeline, cross_validation)

    print(learner.fit(X, y))

if __name__ == '__main__':
    # data = sys.argv[1]
    main()


# import pandas as pd
# import numpy as np
# desired_width = 620
# pd.set_option('display.width', desired_width)
# pd.set_option("display.max_columns", 10)
# np.set_printoptions(linewidth=desired_width)