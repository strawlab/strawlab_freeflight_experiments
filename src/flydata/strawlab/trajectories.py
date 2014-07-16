# coding=utf-8
"""Trajectories (time-series of stimuli and responses) are the object of study of freeflight analysis.
(at least for now)
"""
from itertools import izip
import os.path as op
import datetime

from pandas import DataFrame
import h5py
import time
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
    start_frame: int
        the frame number of the first measurement of a trajectory
    start_time: int
        unix timestamp (UTC) for the first measurement of the trajectory (i.e. when tracking started)
    condition: string
        the string identifying the stimulus
    series: dataframe
        a pandas dataframe with the time series data of the possitions and possibly
        other derived or stimulus series
    dt: float
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
    # Persistence and I/O
    #####################

    @staticmethod
    def to_npz(npz, trajs):
        # This is quite fast, but not really standard / interoperable (and might fail because of zip file limits)
        ensure_dir(op.dirname(npz))
        np.savez(npz,
                 mds=[traj.md().to_json_string() for traj in trajs],
                 oids=[traj.oid() for traj in trajs],
                 start_frames=[traj.start_frame() for traj in trajs],
                 start_times=[traj.start_times() for traj in trajs],
                 conditions=[traj.condition() for traj in trajs],
                 dts=[traj.dt() for traj in trajs],
                 columns=[traj.series().columns for traj in trajs],
                 **{'%d' % i: traj.series() for i, traj in enumerate(trajs)})
            # FIXME: if all (mds, columns, conditions, dts) are the same, then save only one...
            #        would lead to a simple relational schema
            # FIXME: save md as a named tuple, instead of our own class
            # FIXME: save index of DF if it is timedate (although we should be able to reconstruct it)

    @staticmethod
    def from_npz(npz):
        loader = np.load(npz)
        trajs = []
        for i, (md, oid, cond, start, dt, columns) in enumerate(izip(loader['mds'],
                                                                     loader['oids'],
                                                                     loader['conditions'],
                                                                     loader['starts'],
                                                                     loader['dts'],
                                                                     loader['columns'])):
            md = FreeflightExperimentMetadata.from_json_string(md)  # Buggy, probably
            series = DataFrame(loader[str(i)], columns=columns)
            trajs.append(FreeflightTrajectory(md, oid, start, cond, series, dt=dt))
        return trajs

    def to_h5(self, h5, save_experiment_metadata=True, save_trajectory_metadata=True, save_series=True):

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


if __name__ == '__main__':

    # N.B. /tmp is a ramdisk

    import pandas as pd

    md = FreeflightExperimentMetadata(uuid='FAKE_UUID_HERE_AND_THERE',
                                      dictionary=dict(
                                          uuid='FAKE_UUID_HERE_AND_THERE',
                                          user='rudolph',
                                          hidden=False,
                                          tags=('wait', 'caliphora', 'final_paper_experiment'),
                                          title='rotation',
                                          description='some more games on place',
                                          genotype='VT37804-TNTE',
                                          age=4,
                                          arena='flycave',
                                          num_flies=20,
                                          start_secs=7777777777,
                                          start_nsecs=10,
                                          stop_secs=8777777777,
                                          stop_nsecs=10,
                                      ))

    start = time.time()
    trajs = []
    for oid in xrange(10000):
        rng = np.random.RandomState(0)

        df = pd.DataFrame()
        numobs = 1000
        df['x'] = rng.uniform(size=numobs)
        df['y'] = rng.uniform(size=numobs)
        df['z'] = rng.uniform(size=numobs)
        df['rotation_rate'] = rng.uniform(size=numobs)
        df['dtheta'] = rng.uniform(size=numobs)

        traj = FreeflightTrajectory(md, oid, 100, 7777777887, condition='cool|1|2.5|blah.osg|23', series=df, dt=0.1)

        trajs.append(traj)

    print 'Generation took: %.2f' % (time.time() - start)

    start = time.time()
    with h5py.File('/tmp/test.h5') as h5:
        trajs[0].to_h5(h5)
        for traj in trajs[1:]:
            traj.to_h5(h5, save_experiment_metadata=False, save_trajectory_metadata=False)
    print 'Write took: %.2f' % (time.time() - start)

    start = time.time()
    with h5py.File('/tmp/test.h5', 'r') as h5:
        exp_group = h5[u'uuid=FAKE_UUID_HERE_AND_THERE']
        traj_ids = exp_group.keys()
        for traj_id in traj_ids:
            traj_group = exp_group[traj_id]
            series_group = traj_group['series']
            df = DataFrame({series_id: series_group[series_id].value for series_id in series_group})
    print 'Reading from non-contiguous series took: %.2f' % (time.time() - start)

    # Naive contiguous - upper bound of speed
    X = np.random.uniform(size=(10000, 5000))
    start = time.time()
    with h5py.File('/tmp/test-fastest.h5') as h5:
        h5['X'] = X
    print 'Write contiguous: %.2f' % (time.time() - start)

    start = time.time()
    with h5py.File('/tmp/test-fastest.h5') as h5:
        X = h5['X'].value
        print X.shape
    print 'Read contiguous: %.2f' % (time.time() - start)



#### Slow as hell
# Generation took: 14.15
# Write took: 17.36
# Reading from non-contiguous series took: 18.90
# Write contiguous: 0.14
# (10000, 5000)
# Read contiguous: 0.09
#### /Slow as hell
# For not trading with flexibility, we will need to compactify...
####
