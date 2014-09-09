# coding=utf-8
from pandas import DataFrame
from flydata.strawlab.trajectories import FreeflightTrajectory
import numpy as np
from oscail.common.config import Configurable


class FeatureExtractor(Configurable):

    def fnames(self):
        return [self.configuration().id()]

    def compute_one(self, x):
        if isinstance(x, FreeflightTrajectory):
            x = x.series()
        if not isinstance(x, DataFrame):
            raise Exception('We are expecting a pandas DataFrame')
        return self._compute_from_df(x)

    def _compute_from_df(self, df):
        raise NotImplementedError()

    def compute(self, X):
        return np.array([self.compute_one(x) for x in X])


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
