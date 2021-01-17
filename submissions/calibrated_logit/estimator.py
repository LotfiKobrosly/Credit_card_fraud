import numpy as np
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.pipeline import Pipeline


def get_estimator():
    logit = LogisticRegression(C=0.01)
    cal = CalibratedClassifierCV(logit, cv=5)
    renn = RepeatedEditedNearestNeighbours(n_neighbors=20, n_jobs=-1)
    pipeline = Pipeline([('renn', renn), ('cal_logit', cal)])
    return pipeline
