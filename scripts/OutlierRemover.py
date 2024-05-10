from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor

    def _outlier_detector(self, X, y=None):
        X = pd.Series(X).copy()
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1
        self.lower_bound.append(q1 - (self.factor * iqr))
        self.upper_bound.append(q3 + (self.factor * iqr))

    def fit(self, X: np.ndarray, y=None):
        # if fit in another dataset, this two arrays must be reset
        self.lower_bound = []
        self.upper_bound = []

        X.apply(self._outlier_detector)
        self.feature_names_in_ = X.shape[
            1
        ]  # this is required for get_feature_names_out
        return self

    def transform(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X).copy()  # convert X into Pandas dataframe to use .iloc[:, i]
        for i in range(X.shape[1]):
            x = X.iloc[:, i].copy()
            x[(x < self.lower_bound[i]) | (x > self.upper_bound[i])] = np.nan
            X.iloc[:, i] = x
        self.columns = X.columns
        return X

    def get_feature_names_out(self, feature_names):
        return [col for col in self.columns]
