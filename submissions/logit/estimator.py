import numpy as np
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.pipeline import Pipeline


def get_estimator():
    logit = LogisticRegression(C=0.01)
    renn = RepeatedEditedNearestNeighbours(n_neighbors=20, n_jobs=-1)
    pipeline = Pipeline([('renn', renn), ('logit', logit)])
    return pipeline
