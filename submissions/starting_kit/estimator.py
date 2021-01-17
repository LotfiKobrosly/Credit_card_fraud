import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from imblearn import over_sampling, under_sampling


class Uniform(BaseEstimator): # dummy estimator to test ramp
    def __init__(self, value=0):
        self.value = value

    def fit(self, X, y):
        assert len(X) == len(y), f"Wrong dimensions (len(X): {len(X)}, len(y): {len(y)})."
        return self
    
    def predict(self, X):
        y_pred = np.ones((X.shape[0],)) * self.value
        return np.array([1 - y_pred, y_pred]).T


def get_estimator():
    estimator = Uniform(value=1)
    pipeline = Pipeline(steps=[('estimator', estimator)])
    return pipeline
