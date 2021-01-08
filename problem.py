import os

import numpy as np
import pandas as pd
import rampwf as rw
from rampwf.score_types.base import BaseScoreType
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error
