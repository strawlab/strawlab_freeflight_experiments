# coding=utf-8
"""Trajectories (time-series of stimuli and responses) are the object of study of freeflight analysis.
(at least for now)
"""
from itertools import izip
import os.path as op
import datetime

from pandas import DataFrame
import h5py
import numpy as np
import pytz

from flydata.misc import ensure_dir
from flydata.strawlab.metadata import FreeflightExperimentMetadata


class FreeflightTrajectory(object):
    """Enables (in-memory consistent and localized) access to a single trajectory and its associated metadata.

    A trajectory is uniquely identified by the triplet (uuid, oid, start_frame) within the freeflight infrastructure.

    Parameters
    ----------
    md: FreeflightMetadata
        the experiment-level metadata for the trajectory
        (uuid, genotype, experimental setup, researcher...)
    oid: int
        the object-id associated to this trajectory within the scope of the experiment
        this comes from flydra world, where a trajectory can span several trials
        together with start_frame, this is a unique identifier of a trial within the experiment
    start_frame: int
        the frame number of the first measurement of the trial
        together with oid, this is a unique identifier of a trial within the experiment
    frames_before: int, default 0
        how many frames in the time series data correspond to measurements before the tracking start
        * TODO: get rid of this, make it instead a different dataframe *
    start_time: int
        unix timestamp (UTC) for the first measurement of the trial
        (will correspond to when tracking started only if frames_before is zero)
    condition: string
        the string identifying the stimulus
    series: dataframe
        a pandas dataframe with the time series data of the possitions and possibly
        other derived or stimulus series
    dt: float, default 0.01 (100Hz)
        the sampling period in seconds
    """

    __slots__ = ['_md', '_oid', '_start_frame', '_start_time', '_condition', '_dt', '_series']

    def __init__(self, md, oid, start_frame, start_time, condition, series, dt=0.01):
        super(FreeflightTrajectory, self).__init__()
        self._md = md
        self._oid = oid
        self._start_frame = start_frame
        # All the rest might be accessory, since uuid+oid+startframe should be a unique identifier for a trajectory
        self._condition = condition
        self._start_time = start_time
        self._dt = dt
        self._series = series

    def md(self):
        """Returns the metadata for the experiment this trajectory belongs to (see FreeflightMetadata)."""
        return self._md

    def uuid(self):
        """Returns the uuid of the whole experiment this trajectory belongs to."""
        return self._md.uuid()

    def oid(self):
        """Returns the object-id assigned to this trajectory inside the experiment."""
        return self._oid

    def start_frame(self):
        """Returns the number of the first analyzed frame of the trajectory."""
        return self._start_frame

    def id(self):
        """Returns (uuid, oid, framenumber0).
        This is the unique identifier for a trajectory in the whole freeflight infrastructure.
        """
        return self.uuid(), self.oid(), self.start_frame()

    def id_string(self, verbose=True):
        """Returns the  trajectory id as a string.
        One of the two:
          'uuid=blah#oid=bleh#start=blih' (if verbose is True)
          'uuid#oid#start' (if verbose is False)
        This is the unique identifier for a trajectory in the whole freeflight infrastructure.

        """
        if verbose:
            return 'uuid=%s#oid=%d#start=%d' % self.id()
        return '#'.join(map(str, self.id()))

    def genotype(self):
        """Returns the genotype string."""
        return self._md.genotype()  # N.B. we assume an experiment only spans one genotype
                                    # rework if that is not the case

    def condition(self):
        """Returns the stimulus description string."""
        return self._condition  # N.B. we assume a trajectory spans only one condition
                                # rework if that is not the case

    def dt(self):
        """Returns the sampling period, in seconds."""
        return self._dt

    def start_time(self):
        """Returns the unix timestamp (UTC) for the first measurement of the trajectory (when tracking started)."""
        return self._start_time

    def start_asdatetime(self, tzinfo=pytz.timezone('Europe/Vienna')):
        """Returns the start time as a python timestamp for the specified time-zone."""
        return datetime.datetime.fromtimestamp(self.start_time(), tz=tzinfo)

    def start_asisoformat(self, tzinfo=pytz.timezone('Europe/Vienna')):
        """Returns the start time as a string in ISO 8601 format."""
        return self.start_asdatetime(tzinfo=tzinfo).isoformat()

    def series(self, columns=None, copy=False):
        """Returns a pandas dataframe with the series data for the trajectory.
        If copy is True, a deepcopy of the dataframe is returned.
        """
        if callable(self._series):
            self._series = self._series()  # Lame laziness, forces to load everything in memory
        if columns is None:
            return self._series.copy() if copy else self._series
        else:
            df = self.series(copy=copy)
            columns = [coord for coord in columns if coord in df.columns]
            return df[columns]
            # TODO: implement laziness, _series should not be assumed a pandas dataframe

    def df(self, copy=False):
        """An alias to series()."""
        return self.series(copy=copy)

    def xyz(self, copy=False):
        """Returns  pandas dataframe the tracked position of the fly, (usually after forward-backward smoothing).
        This returns the x, y and z columns from the series() dataframe (if present).
        Everything else there would have been computed from x,y,z and possibly stimulus data.
        """
        return self.series(copy=copy, columns=('x', 'y', 'z'))

    ####################
    # Official mutability breakers
    ####################

    def set_series(self, series):
        # FIXME: workaround for wrong design
        self._series = series

    def apply_transform_inplace(self, **kwargs):
        """Applies single-attribute transformations to this trajectory.

        This method endulges responsible API users with a way to break immutability by "official ways".

        Parameters
        ----------
        kwargs: pairs attribute_name -> transform
            - Each attribute name should correspond to the name of an attribute in the trajectory
              (otherwise an exception is raised)
            - transform is a function that takes a trajectory and returns a single value;
              such value substitutes the attribute's previous value

        Examples
        --------
        If "traj" is a trajectory, then this call will redefine the "dt" and "condition" attributes:
          traj.apply_transform_inplace(
              dt=lambda traj: traj.dt() * 10,
              condition=lambda traj: traj.condition() + '-NORMALIZED_CONDITION'
          )
        """
        for key, fix in kwargs.iteritems():
            attr = '_%s' % key
            if not attr in self.__slots__:
                raise Exception('The key %s is not known to our trajectory (sorry, we are kinda unflexible ATM)' % key)
            setattr(self, attr, fix(self))  # Awful

    #####################
    #
    # Deep equality test.
    # Even if we actually only need __eq__, thios got me busy reading about Custom rich comparisons support
    #   - http://legacy.python.org/dev/peps/pep-0207/
    #   - http://regebro.wordpress.com/2010/12/13/python-implementing-rich-comparison-the-correct-way/
    #   - http://www.voidspace.org.uk/python/articles/comparison.shtml
    #
    #####################


    #####################
    # Persistence and I/O
    #####################

    @staticmethod
    def to_npz(npz,
               trajs,
               compress_mds=True,
               compress_conditions=True,
               compress_columns=False,
               compress_dt=False,
               save_index=False):
        if compress_dt:
            raise NotImplementedError('Compressing dt (assume single value) not implemented at the moment')
        if compress_columns:
            raise NotImplementedError('Compressing columns not implemented at the moment')
        if save_index is True:
            raise NotImplementedError('Pandas Index saving not implemented at the moment')
        # This is quite fast, but not really standard / interoperable (and might fail because of zip file limits)
        # Of course, no memcache - each trajectory series would need to be on its own file
        ensure_dir(op.dirname(npz))
        # Metadatas
        mds = trajs[0].md().to_json_string() if compress_mds else [traj.md().to_json_string() for traj in trajs]
        # Conditions
        conditions = [traj.condition() for traj in trajs]
        present_conditions = sorted(set(conditions))
        if compress_conditions:
            conditions_dict = {condition: i for i, condition in enumerate(present_conditions)}
            conditions = [conditions_dict[condition] for condition in conditions]
        def remove_non_numeric_columns(df):
            return df._get_numeric_data()  # We are all adults here
        np.savez(npz,
                 num_trajs=len(trajs),
                 mds=mds,
                 oids=[traj.oid() for traj in trajs],
                 start_frames=[traj.start_frame() for traj in trajs],
                 start_times=[traj.start_time() for traj in trajs],
                 present_conditions=present_conditions,
                 conditions=conditions,
                 dts=[traj.dt() for traj in trajs],
                 columns=[map(str, remove_non_numeric_columns(traj.series()).columns) for traj in trajs],
                 **{'%d' % i: remove_non_numeric_columns(traj.series()).values for i, traj in enumerate(trajs)})
            # We are rolling our own simple relational schema...
            # FIXME: save index of DF if it is timedate
            #        (although we should be able to reconstruct it if no holes are present, check...)

    @staticmethod
    def from_npz(npz,
                 md=None,
                 traj_ids=None,
                 lazy_series=False):
        if traj_ids is not None:
            raise NotImplementedError('Cannot read individual trajectories at the moment')
            # easy to implement, but requires index
        if lazy_series is True:
            raise NotImplementedError('Cannot apply lazy loading of series at the moment')
        loader = np.load(npz)
        trajs = []
        num_trajs = loader['num_trajs'].item()
        # Metadata
        if isinstance(md, FreeflightExperimentMetadata):
            mds = [md] * num_trajs
        elif md is None:
            mds = loader['mds']
            if mds.ndim == 0:
                mds = [FreeflightExperimentMetadata.from_json_string(mds.item())] * num_trajs
            elif len(mds) != num_trajs:
                raise Exception('There is something wrong assigning metadata to each ')
        else:
            raise Exception('md must be either None or a single FreeflightExperimentMetadata object')
        # Conditions
        conditions = loader['conditions']
        if not isinstance(conditions.dtype, basestring):
            present_conditions = loader['present_conditions']
            conditions = [present_conditions[i] for i in conditions]
        # Read all trajectories
        for i, (md, oid, cond, start_frame, start_time, dt, columns) in enumerate(izip(mds,
                                                                                       loader['oids'],
                                                                                       conditions,
                                                                                       loader['start_frames'],
                                                                                       loader['start_times'],
                                                                                       loader['dts'],
                                                                                       loader['columns'])):
            series = DataFrame(loader['%d' % i], columns=columns)
            trajs.append(FreeflightTrajectory(md, oid, start_frame, start_time, cond, series, dt=dt))
        return trajs

    @staticmethod
    def to_h5(h5,
              trajs,
              root_group='/',
              compress_mds=True,
              compress_conditions=True,
              compress_columns=False,
              compress_dt=False,
              save_index=False):
        if compress_dt:
            raise NotImplementedError('Compressing dt (assume single value) not implemented at the moment')
        if compress_columns:
            raise NotImplementedError('Compressing columns not implemented at the moment')
        if save_index is True:
            raise NotImplementedError('Pandas Index saving not implemented at the moment')
        # This is quite fast, but not really standard / interoperable (and might fail because of zip file limits)
        # Of course, no memcache - each trajectory series would need to be on its own file
        ensure_dir(op.dirname(h5))
        # Metadatas
        mds = trajs[0].md().to_json_string() if compress_mds else [traj.md().to_json_string() for traj in trajs]
        # Conditions
        conditions = [str(traj.condition()) for traj in trajs]  # h5py (hdf5) does not support unicode
        present_conditions = sorted(set(conditions))
        if compress_conditions:
            conditions_dict = {condition: i for i, condition in enumerate(present_conditions)}
            conditions = [conditions_dict[condition] for condition in conditions]
        def save_in_h5(h5, **kwargs):
            for k, v in kwargs.iteritems():
                h5[k] = v
        def remove_non_numeric_columns(df):
            return df._get_numeric_data()  # We are all adults here
        with h5py.File(h5, 'w-') as h5:
            group = h5[root_group]
            save_in_h5(group,
                       num_trajs=len(trajs),  # Should better be an attribute of the dataset
                       mds=mds,
                       oids=[traj.oid() for traj in trajs],
                       start_frames=[traj.start_frame() for traj in trajs],
                       start_times=[traj.start_time() for traj in trajs],
                       present_conditions=present_conditions,
                       conditions=conditions,
                       dts=[traj.dt() for traj in trajs],
                       columns=[map(str, remove_non_numeric_columns(traj.series()).columns) for traj in trajs],
                       **{'%d' % i: remove_non_numeric_columns(traj.series()) for i, traj in enumerate(trajs)})
            # TODO: make explicit that we remove non-numeric columns and do it only once
            #       also, it is necessary just because we are dealing with some legacy pickles
            # TODO: we could just use pandas and save that many datasets with attributes
            # TODO: we could just make attributes everything but the series
            # We are rolling our own simple relational schema...
            # FIXME: save index of DF if it is timedate
            #        (although we should be able to reconstruct it if no holes are present, check...)

    @staticmethod
    def from_h5(h5,
                md=None,
                traj_ids=None,
                lazy_series=False):
        if traj_ids is not None:
            raise NotImplementedError('Cannot read individual trajectories at the moment')
            # easy to implement, but requires index
        if lazy_series is True:
            raise NotImplementedError('Cannot apply lazy loading of series at the moment')
        with h5py.File(h5, 'r') as loader:
            trajs = []
            num_trajs = loader['num_trajs'][()]
            # Metadata
            if isinstance(md, FreeflightExperimentMetadata):
                mds = [md] * num_trajs
            elif md is None:
                mds = loader['mds'][:]
                if mds.ndim == 0:
                    mds = [FreeflightExperimentMetadata.from_json_string(mds.item())] * num_trajs
                elif len(mds) != num_trajs:
                    raise Exception('There is something wrong assigning metadata to each ')
            else:
                raise Exception('md must be either None or a single FreeflightExperimentMetadata object')
            # Conditions
            conditions = loader['conditions'][:]
            if not isinstance(conditions.dtype, basestring):
                present_conditions = loader['present_conditions'][:]
                conditions = [present_conditions[i] for i in conditions]
            # Read all trajectories
            for i, (md, oid, cond, start_frame, start_time, dt, columns) in enumerate(izip(mds,
                                                                                           loader['oids'],
                                                                                           conditions,
                                                                                           loader['start_frames'],
                                                                                           loader['start_times'],
                                                                                           loader['dts'],
                                                                                           loader['columns'])):
                series = DataFrame(data=loader['%d' % i][:], columns=columns)
                trajs.append(FreeflightTrajectory(md, oid, start_frame, start_time, cond, series, dt=dt))
            return trajs

    def to_h5_identifiable(self, h5, save_experiment_metadata=True, save_trajectory_metadata=True, save_series=True):

        # Save experiment metadata
        exp_group = h5.require_group('uuid=%s' % self.uuid())
        if save_experiment_metadata:
            attrs = exp_group.attrs
            for k, v in self.md().flatten():
                attrs[k] = v

        # Save trajectory metadata
        traj_group = exp_group.require_group('oid=%d#frame=%d' % (self.oid(), self.start_frame()))
        if save_trajectory_metadata:
            attrs = traj_group.attrs
            attrs['condition'] = self.condition()
            attrs['start_time'] = self.start_time()
            # TODO: assess using a nominal attr for condition; a possible implementation would resort toh5py.SoftLink

        # Save the series
        if save_series:
            series_group = traj_group.require_group('series')
            series = self.series()
            for column in series.columns:
                series_group[column] = series[column]
        # TODO: API for lazy access, do not assume pandas dataframe as the representation (or use a lazy dataset)
        # TODO: implement index saving, using softlinks when needed

    @staticmethod
    def to_pandas(trajs, flatten=True):
        """Represent a collection of trajectories as a pandas data-frame, record format."""
        if not flatten:
            return DataFrame(data=trajs, columns=('trajectories',))
        columns = ('uuid', 'oid', 'genotype', 'condition', 'dt', 'start', 'series')
        return DataFrame(data=((traj.uuid(),
                               traj.oid(),
                               traj.genotype(),
                               traj.condition(),
                               traj.dt(),
                               traj.start(),
                               traj.series()) for traj in trajs),
                         columns=columns,
                         dtype=(np.str, np.int, np.str, np.str, np.float, np.float, object))

    @staticmethod
    def from_pandas(df, flattened=True):
        raise NotImplementedError()


###################
# Convenience functions
###################

def df_or_df_from_traj(df):
    """Returns a pandas data-frame, either df itself or the series associated to a trajectory object."""
    # Problem: well against duck typing
    if isinstance(df, FreeflightTrajectory):
        return df.series()
    if isinstance(df, DataFrame):
        return df
    raise Exception('The type of the variable (%r) is not one of FreeflightTrajectory or DataFrame, '
                    'and this is kinda static python'
                    % type(df))