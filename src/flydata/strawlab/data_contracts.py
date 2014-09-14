# coding=utf-8
import numpy as np
from pandas import DataFrame
from pandas.util.testing import isiterable
from flydata.strawlab.trajectories import FreeflightTrajectory


def df_or_df_from_traj(df):
    # Problem: well against ducks
    if isinstance(df, FreeflightTrajectory):
        return df.series()
    if isinstance(df, DataFrame):
        return df
    raise Exception('The type of the variable (%r) is not one of FreeflightTrajectory or DataFrame, '
                    'and this is kinda static python'
                    % type(df))


class DataContract(object):

    def check(self, X):
        """
        Checks the data in X, returning
        """
        raise NotImplementedError()

    def format_report(self, report):
        raise NotImplementedError()


class NoMissingValuesContract(DataContract):

    def __init__(self, columns=('x', 'y', 'z')):
        super(NoMissingValuesContract, self).__init__()
        self.columns = columns

    def check(self, X):
        if not isiterable(X):
            X = [X]
        for i, traj in enumerate(X):
            traj = df_or_df_from_traj(traj)
            # or use traj[self.columns].isnull() and relate to columns afterwards...
            missing_rows = {col: np.where(traj[col].isnull())[0] for col in self.columns}
            yield not any(len(rows) > 0 for rows in missing_rows.values()), missing_rows


class NoHolesContract(DataContract):

    def check(self, X):
        pass