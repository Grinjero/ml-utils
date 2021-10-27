# from sklearn.base import BaseEstimator
from pmdarima.preprocessing.base import BaseTransformer
import numpy as np


class Differentiator(BaseTransformer):
    def __init__(self, differencing_period):
        self.differencing_period = differencing_period

    def fit(self, y, X=None, **fit_args):
        pass

    def transform(self, y, X=None, **kwargs):
        print(y)
        print(type(y))

        return y, X