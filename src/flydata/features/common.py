# coding=utf-8
from pandas import DataFrame
import numpy as np

from flydata.strawlab.trajectories import df_or_df_from_traj
from oscail.common.config import Configurable
import pandas as pd


# TODO: make FeatureExtractor also follow the fit-transform protocol
# TODO: make SeriesExtractor not to work inplace (at least by default)

class FeatureExtractor(Configurable):

    def fnames(self):
        return [self.configuration().id()]

    def compute_one(self, x):
        x = df_or_df_from_traj(x)
        if not isinstance(x, DataFrame):
            raise Exception('We are expecting a pandas DataFrame')
        return self._compute_from_df(x)

    def _compute_from_df(self, df):
        raise NotImplementedError()

    def compute(self, X):
        return np.array([self.compute_one(x) for x in X])


def compute_features(trajs, feature_extractors):
    """A loop to compute a set of scalar features for trajectories."""
    fnames = []
    fvalues = []
    for feature in feature_extractors:
        fnames += feature.fnames()
        fvalues.append(feature.compute(trajs))
    return pd.DataFrame(data=np.vstack(fvalues).T, columns=fnames, index=[traj.id_string() for traj in trajs])


class SeriesExtractor(FeatureExtractor):  # probably we can unify with FeatureExtractor

    def compute(self, X):
        # Nasty with the current semantics as we always change the df in place
        # So return type changes from FeatureExtractor
        for x in X:
            self.compute_one(x)
        return X


class Length(FeatureExtractor):

    def __init__(self, column='x'):
        super(Length, self).__init__()
        self.column = column

    def _compute_from_df(self, df):
        return len(df[self.column])


class Mean(FeatureExtractor):

    def __init__(self, column):
        super(Mean, self).__init__()
        self.column = column

    def _compute_from_df(self, df):
        return df[self.column].mean()


class Std(FeatureExtractor):

    def __init__(self, column):
        super(Std, self).__init__()
        self.column = column

    def _compute_from_df(self, df):
        return df[self.column].std()


class Kurtosis(FeatureExtractor):

    def __init__(self, column):
        super(Kurtosis, self).__init__()
        self.column = column

    def _compute_from_df(self, df):
        return df[self.column].kurtosis()


class Skew(FeatureExtractor):

    def __init__(self, column):
        super(Skew, self).__init__()
        self.column = column

    def _compute_from_df(self, df):
        return df[self.column].skew()
