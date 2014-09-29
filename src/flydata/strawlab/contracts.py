# coding=utf-8
import numpy as np
from pandas import DatetimeIndex
from pandas.util.testing import isiterable

from flydata.strawlab.trajectories import df_or_df_from_traj
from oscail.common.config import Configurable

__all__ = (
    # superclass
    'DataContract',
    # contracts
    'NoMissingValuesContract',
    'NoHolesContract',
    'AllNumericContract',
    # functions
    'check_contracts'
)


class DataContract(Configurable):

    def __init__(self, fail_message=None):
        super(DataContract, self).__init__()
        self._fail_message = fail_message if fail_message is not None else '%s failed' % self.__class__.__name__

    def check(self, X):
        """Checks the data in X, returning True iff all the data points agree with the contract."""
        if not isiterable(X):
            X = [X]
        for x in X:
            if not self._check_one(x):
                return False
        return True

    def _check_one(self, x):
        raise NotImplementedError()

    ########################
    # Report generation
    ########################

    def check_and_report(self, X):
        """Returns a generator; each element in the generator is a tuple (agree, report_data) for each object in X."""
        return ((self.check(x), None) for x in ([X] if not isiterable(X) else [X]))

    def format_report(self, report_data):
        return ''

    def check_all_and_format_report(self, X):
        return '\n'.join(['%d: %r %s' % (i, agree, self.format_report(report_data))
                          for i, (agree, report_data) in enumerate(self.check_and_report(X))])

    @property
    def fail_message(self):
        return self._fail_message


############################################
# Missing values
############################################

class NoMissingValuesContract(DataContract):

    def __init__(self, columns=None):
        super(NoMissingValuesContract, self).__init__('Some columns have missing values, but we do not support them')
        self.columns = columns

    def _check_one(self, x):
        df = df_or_df_from_traj(x)
        for column in self.columns if self.columns is not None else df.columns:
            if df[column].isnull().any():
                return False
        return True

    def check_and_report(self, X):
        if not isiterable(X):
            X = [X]
        for traj in X:
            df = df_or_df_from_traj(traj)
            # or use traj[self.columns].isnull() and relate to columns afterwards...
            missing_rows = {col: np.where(df[col].isnull())[0]
                            for col in (self.columns if self.columns is not None else df.columns)}
            yield not any(len(rows) > 0 for rows in missing_rows.values()), missing_rows

    def format_report(self, report_data):
        return 'Missing Values: %s' % ';'.join(['%s:%s' % (col, str(missing_rows))
                                                for col, missing_rows in report_data.iteritems()
                                                if len(missing_rows) > 0])


############################################
# No holes in data-frames
############################################

def has_holes(df):
    observations_distances = df.index.values[1:] - df.index.values[0:-1]
    return not all(1 == observations_distances)


class NoHolesContract(DataContract):

    def __init__(self):
        super(NoHolesContract, self).__init__('Some columns have holes, but we do not support them')

    def _check_one(self, x):
        x = df_or_df_from_traj(x)
        if isinstance(x.index, DatetimeIndex):  # Works for all cases we have now, but make this more general
            raise Exception('pandas Datetime indices are not supported at the moment')
        return not has_holes(x)


############################################
# All the series have numeric values only
############################################

class AllNumericContract(DataContract):

    def __init__(self, columns=None):
        super(AllNumericContract, self).__init__('Some columns are non-numeric, but we do not support them')
        self.columns = columns

    def _check_one(self, x):
        df = df_or_df_from_traj(x)
        for column in self.columns if self.columns is not None else df.columns:
            # Good discussion:
            #   http://stackoverflow.com/questions/500328/identifying-numeric-and-array-types-in-numpy
            if not np.issubdtype(df[column].dtype, np.number):
                return False
        return True


############################################
# Series are monotonically increasing / decreasing
############################################

def non_monotonic_regions(x, increasing=True, strict=False, max_jump=None):
    """
    Looks for regions in a time series that are not monotonic.


    Parameters
    ----------
    x : numpy array

    increasing : boolean, default True
      If True we are looking for non-increasing regions (e.g. [5, 4] will be flagged);
      if False, we are looking for non-decreasing reginons (e.g. [4, 5] will be flagged).

    strict:
      If True, strictly monotonic regions (e.g. [4, 4]) will be flagged;
      if False, strictly monotonic regions (e.g. [4, 4]) will not be flagged.

    discontinuity: number or None, default None
      If it is a number, monotonicity violations that cross the discontinuity value are not
      included in the non-monotonic regions. This is useful for wrapped quantities
      (like angles or stimulus ratio).


    Returns
    -------
    A numpy bool array of the same dimension as x indicating where x is non-monotonic.


    Examples
    --------
    >>> x = [1, 2, 3, 4, 4]
    >>> np.sum(non_monotonic_regions(x, increasing=True, strict=True))
    1
    >>> non_monotonic_regions(x, increasing=True, strict=True)[-1]
    True
    >>> non_monotonic_regions(x, increasing=True, strict=False)
    array([False, False, False, False, False], dtype=bool)
    >>> non_monotonic_regions(x, increasing=False, strict=False)
    array([False,  True,  True,  True, False], dtype=bool)
    >>> x = [0.1, 0.3, 0.25, 0.8, 1.0, 0.1]
    >>> non_monotonic_regions(x, increasing=True, strict=False, max_jump=0.5)
    array([False, False,  True, False, False, False], dtype=bool)
    """
    # Vectorized, not specially efficient version
    # FIXME: Discontinuity treatment by max_jump is error prone, also doc is now not correct
    xd = np.diff(x)
    if increasing:
        nonmonotonical = (xd <= 0) if strict else (xd < 0)
    else:
        nonmonotonical = (xd >= 0) if strict else (xd > 0)
    if max_jump is not None:
        nonmonotonical &= (np.abs(xd) < max_jump)
    return np.insert(nonmonotonical, 0, False)


class MonotonicalContract(DataContract):

    def __init__(self, columns=None, increasing=True, strictly=False, max_jump=None):
        super(MonotonicalContract, self).__init__('Some columns are not monotonical, but we do not support them')
        self.columns = columns
        self.increasing = increasing
        self.strictly = strictly
        self.max_jump = max_jump

    def _check_one(self, x):
        df = df_or_df_from_traj(x)
        for column in self.columns if self.columns is not None else df.columns:
            if non_monotonic_regions(df[column].values,
                                     increasing=self.increasing,
                                     strict=self.strictly,
                                     max_jump=self.max_jump).any():
                return False
        return True


############################################
# Convenience functions
############################################

def check_contracts(trajs, contracts):
    for contract in contracts:
        if not contract.check(trajs):
            raise Exception(contract.fail_message)
    return [contract.who().id() for contract in contracts]
