# coding=utf-8
import numpy as np
from abc import ABCMeta, abstractmethod
from flydata.strawlab.trajectories import df_or_df_from_traj
from oscail.common.config import Configurable

__all__ = (
    'Transformer',
    'ColumnsSelector',
    'NumericEnforcer',
    'MissingImputer',
)


class Transformer(Configurable):
    """
    A filter class transform a collection of objects in some way.
    As it is common in data-analysis pipelines, we use a two phase protocol (and model the API after sklearn)

      1. Filter.fit(X, y=None) will "train" the filter using data X and possibly also labels y
                               for example, compute max and mins for normalisation

      2. Filter.transform(X) will apply the fitted transformation to X (and possibly y, different from sklearn)

    This should capture also most of our use-cases.
    Here, our objects will probably be either time-series or feature matrices.
    """

    __metaclass__ = ABCMeta

    __slots__ = ()  # So that we allow subclasses to decide if they want slots or not

    def __init__(self, add_descriptors=False):
        super(Transformer, self).__init__(add_descriptors)

    def fit(self, X, y=None):
        """Computes anything needed to apply the filter given X and possibly y."""
        self._fit_hook(X, y=y)
        return self  # Do nothing by default, but allow fluency

    def _fit_hook(self, X, y=None):
        pass

    @abstractmethod
    def transform(self, X):
        """Returns transformed X. In many instances, "fit" should be called first."""
        raise NotImplementedError()  # abc

    def fit_transform(self, X, y=None):
        """Fits the transformer and applies the transformation to X, returning transformed X."""
        return self.fit(X, y=y).transform(X)


##############################
# Select series from dataframe
##############################


class ColumnsSelector(Transformer):

    def __init__(self, series_to_keep):
        super(ColumnsSelector, self).__init__()
        self.keepers = sorted(series_to_keep)
        self._keepers_set = set(self.keepers)

    def transform(self, trajs):
        for traj in trajs:
            # traj.set_series(traj.series()[self.keepers])  LAME 1
            df = df_or_df_from_traj(traj)
            to_drop = set(df.columns) - self._keepers_set
            df.drop(to_drop, axis=1, inplace=True)  # Lame inplace,
                                                    # because we do not force traj to be FreeflightTrajectory...

        return trajs


##############################
# Force numeric series
##############################

class NumericEnforcer(Transformer):

    def __init__(self, columns=None):
        super(NumericEnforcer, self).__init__()
        self.columns = columns

    def transform(self, X):
        for x in X:
            df = df_or_df_from_traj(x)
            for column in self.columns if self.columns is not None else df.columns:
                # Good discussion:
                #   http://stackoverflow.com/questions/500328/identifying-numeric-and-array-types-in-numpy
                if not np.issubdtype(df[column].dtype, np.number):
                    df[column] = df[column].astype(np.float, copy=False)  # Unflexible.
                                                                        # It should be easy to specify per-column types
        return X


##############################
# Missing values imputation
##############################
#
# - TODO: use the "closest observation" (can be done in 1 pass with lookahead)
# - TODO: investigate if we can get faster with pandas
# - TODO: we can terminate early in the second pass as soon as we find a non-missing
# - TODO: play with others and specialise types for numba, but ultimately just use cython / vectorised numpy
#         (remove weird deps for faster computation)
# - TODO: Compare to fillna(method='ffill'), without selecting columns
#
##############################


class MissingImputer(Transformer):
    """
    Transforms (INPLACE!) the trajectories or data-frames by assigning each
    missing value to its previous or next non-missing value.

    This is useful - and needed - as applied over stimulus series
    that contain missing values due to the controller working at
    slower rate than flydra data acquisition. If it is the other way
    around, foward / backward fill strategies are not the most
    appropriate ones.

    Faster version works only on floating point data at the moment.
    """

    def __init__(self,
                 columns=('rotation_rate', 'trg_x', 'trg_y', 'trg_z', 'ratio'),
                 first_pass_method='ffill',
                 second_pass_method='bfill',
                 faster_if_available=True,
                 verbose=False):
        super(MissingImputer, self).__init__()
        self.columns = list(columns)
        self.fpm = first_pass_method
        self.spm = second_pass_method
        self._faster_if_available = faster_if_available
        self._faster_fills = self.faster_fills() if faster_if_available else None
        self._verbose = verbose

    def transform(self, trajs):
        for i, traj in enumerate(trajs):
            if self._verbose:
                print i
            df = df_or_df_from_traj(traj)
            # inplace not working + inplace should not be default
            # cannot make it work like this in 5 minutes...
            #   df.loc[self.series_names].fillna(method=self.fpm, inplace=True)
            #   if self.spm is not None:
            #       df.loc[self.series_names].fillna(method=self.spm, inplace=True)
            # so ugly for:
            if self._faster_fills is None:
                # Cannot make pandas fast here. It Should be, asthis requires only one pass on each col!
                for column in self.columns:
                    df[column] = df[column].fillna(method=self.fpm)  # N.B. inplace is damn slow in pandas 0.14
                    if self.spm is not None:
                        df[column] = df[column].fillna(method=self.spm)
            else:
                for column in self.columns:
                    try:
                        self._faster_fills[self.fpm](df[column].values)
                        if self.spm is not None:
                            self._faster_fills[self.spm](df[column].values)
                    except:
                        print column, df[column].dtype, df[column].iloc[0]

            # FIXME: traj.set_series(df); depending on the final trajectory API, this might be necessary

        return trajs

    @staticmethod
    def faster_fills():
        try:
            import numba

            @numba.autojit
            def ffill(array):
                last = array[0]
                for i in xrange(1, len(array)):
                    if array[i] != array[i]:
                        array[i] = last
                    else:
                        last = array[i]

            @numba.autojit
            def bfill(array):
                last = array[len(array) - 1]
                for i in xrange(1, len(array)):
                    i = len(array) - 1 - i
                    if array[i] != array[i]:
                        array[i] = last
                    else:
                        last = array[i]

            return {
                'ffill': ffill,
                'bfill': bfill
            }

        except:
            print 'WARNING: numba is not installed, slow filling will make you impatient'  # Proper LOG
            return None


##############################
# Missing values removal
##############################

class RowsWithMissingRemover(Transformer):

    def __init__(self, columns=None, log_removed=False):
        super(RowsWithMissingRemover, self).__init__()
        self.columns = columns
        self._log_removed = log_removed

    def transform(self, X):
        kept = []
        for x in X:
            df = df_or_df_from_traj(x)
            columns = self.columns if self.columns is not None else df.columns
            if not df[columns].isnull().any().any():
                kept.append(x)
            elif self._log_removed:
                print '%s has missing values, removed...' % x.id_string()
        return kept


##############################
# Global TODOs
##############################
#
# TODO: we should not make inplace a default but instead an option disabled by default
# TODO: we should assume that we always have metadata, so we can give better feedback and identities on failures
#
##############################

