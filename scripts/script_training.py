__author__ = 'diegopinheiro'
__email__ = 'diegompin@gmail.com'
__github__ = 'https://github.com/diegompin'


from src.algorithm_strategies.random_forest_strategy import RandomForestRandomizedStrategy
from src.training_strategies.training_strategy import TextPipeline
from src.training_strategies.validation_strategies import KFoldCrossValidation


def main():
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
