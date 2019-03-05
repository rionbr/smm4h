'''
SOURCES
https://github.com/queirozfcom/python-sandbox/blob/master/python3/notebooks/pipeline-examples-post/pipeline-text-single-label-grid-search.ipynb
https://scikit-learn.org/stable/tutorial/statistical_inference/putting_together.html
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
'''

import numpy as np

#TODO Separate plotter
import matplotlib.pyplot as plt


import multiprocessing
import itertools as it
from functools import partial

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold


from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics.scorer import roc_auc_score


class AssessmentFramework(object):

    def __init__(self): pass

    def get_clf(self):
        from sklearn.ensemble import RandomForestClassifier
        model_rf = RandomForestClassifier()
        return model_rf

    def get_pipeline(self, clf):
        estimators = list()
        estimators.append(('vect', CountVectorizer()))
        estimators.append(('tfidf', TfidfTransformer()))
        # estimators.append(('standardize', StandardScaler()))
        estimators.append(('clf', clf))
        model_pipeline = Pipeline(estimators)
        return model_pipeline

    def get_cv(self):
        seed = 7
        kfold = KFold(n_splits=10, random_state=seed)
        return kfold

    def get_search(self):
        clf = self.get_clf()
        cv = self.get_cv()
        model_pipeline = self.get_pipeline(clf)
        param_grid = self.get_params()
        #TODO scorer
        search = GridSearchCV(model_pipeline, cv=cv, param_grid=param_grid)
        return search

    def get_params(self):
        param_grid = {
            'clf__n_estimators': [1,10],
            'clf__max_leaf_nodes': [5,10]
        }
        return param_grid

    #TODO
    def get_scores(self, y_test, y_preds):
        scores = dict()
        scores['matthew'] = matthews_corrcoef(y_test, y_preds)
        scores['roc'] = roc_auc_score(y_test, y_preds)
        return scores

    #TODO
    def plot(self, scores):
        from statsmodels.distributions.empirical_distribution import ECDF
        fig, ax = plt.subplots()
        ecdf = ECDF(scores)
        bins = np.linspace(0, 1)
        ax.plot(bins, ecdf(bins))
        plt.show()

    def bootstrap_distribution(self, scores):
        x = scores
        n = len(scores)
        reps = 1000
        xb = np.random.choice(x, (n, reps))
        mb = xb.mean(axis=0)
        mb.sort()
        p_25, p_975 = np.percentile(mb, [2.5, 97.5])
        return mb, p_25, p_975

    def plot_bootstrap(self, mb):
        fig, ax = plt.subplots()
        ax.hist(mb)
        plt.show()



#TODO Change for tweets
def get_data():
    from sklearn.datasets import fetch_20newsgroups
    cats = ['alt.atheism', 'sci.space']
    newsgroups_train = fetch_20newsgroups(subset='train', categories=cats)
    newsgroups_test = fetch_20newsgroups(subset='test', categories=cats)
    X_train = newsgroups_train.data
    X_test = newsgroups_test.data
    y_train = newsgroups_train.target
    y_test = newsgroups_test.target
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = get_data()


framework = AssessmentFramework()

search = framework.get_search()

#TODO
search.fit(X_train[0:30], y_train[0:30])
print("Best: %f using %s" % (search.best_score_,
                             search.best_params_))
means = search.cv_results_['mean_test_score']
stds = search.cv_results_['std_test_score']
params = search.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))



best_estimator = search.best_estimator_
best_estimator.fit(X_train,y_train)
y_preds = best_estimator.predict(X_test)

framework.get_scores(y_test, y_preds)