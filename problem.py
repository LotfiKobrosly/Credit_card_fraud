import os
import numpy as np
import pandas as pd
import rampwf as rw
from sklearn.model_selection import ShuffleSplit


from sklearn.metrics import f1_score, precision_score
from rampwf.score_types.classifier_base import ClassifierBaseScoreType

class F1_score(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='f1', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        f1 = f1_score(y_true, y_pred)
        return f1

class Precision_score(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='pre', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        pre = precision_score(y_true, y_pred)
        return pre


problem_title = 'Credit Card Fraud Detection'
_target_column_name = 'Class'
_ignore_column_names = ['Time']
_prediction_label_names = [0,1]

# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)

# An object implementing the workflow
workflow = rw.workflows.Estimator()

score_types = [
    F1_score(name='f1', precision=5),
    Precision_score(name='pre', precision=5)
]


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=5, test_size=0.20, random_state=57)
    return cv.split(X, y)


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name))
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name] + _ignore_column_names, axis=1)
    return X_df, y_array


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)
