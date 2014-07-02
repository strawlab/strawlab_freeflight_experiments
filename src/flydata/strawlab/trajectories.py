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
        """Return (uuid, oid, framenumber0).
        This is the unique identifier for a trajectory in the whole freeflight infrastructure.
        """
        return self.uuid(), self.oid(), self.start_frame()

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
        return self._start_time()

    def start_asdatetime(self, tzinfo=pytz.timezone('Europe/Vienna')):
        """Returns the start time as a python timestamp for the specified time-zone."""
        return datetime.datetime.fromtimestamp(self.start_time(), tz=tzinfo)

    def start_asisoformat(self, tzinfo=pytz.timezone('Europe/Vienna')):
        """Returns the start time as a string in ISO 8601 format."""
        return self.start_asdatetime(tzinfo=tzinfo).isoformat()

    def series(self, copy=False):
        """Returns a pandas dataframe with the series data for the trajectory.
        If copy is True, a deepcopy of the dataframe is returned.
        """
        return self._series.copy() if copy else self._series

    def df(self, copy=False):
        """An alias to series()."""
        return self.series(copy=copy)

    def xyz(self, copy=False):
        """Returns  pandas dataframe the tracked position of the fly, (usually after forward-backward smoothing).
        This returns the x, y and z columns from the series() dataframe (if present).
        Everything else there would have been computed from x,y,z and possibly stimulus data.
        """
        df = self.series(copy=copy)
        columns = [coord for coord in ('x', 'y', 'z') if coord in df.columns]
        return df[columns]

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

    @staticmethod
    def to_h5(h5file, trajs, overwrite=False):
        ensure_dir(op.dirname(h5file))
        with h5py.File(h5file, 'w' if overwrite else 'a') as h5:
            h5['mds'] = [traj.md() for traj in trajs]
            h5['oids'] = [traj.oid() for traj in trajs]
            h5['conditions'] = [traj.condition() for traj in trajs]
            h5['starts'] = [traj.start() for traj in trajs]
            h5['dts'] = [traj.dt() for traj in trajs]
            h5['columns'] = [traj.series().columns for traj in trajs]
            for i, traj in enumerate(trajs):
                h5['%d' % i] = traj.series().values

    @staticmethod
    def from_h5(h5file):
        # FIXME: Testme
        import h5py
        with h5py.File(h5file, 'r') as h5:
            trajs = []
            for i, (md, oid, cond, start, dt, columns) in enumerate(izip(h5['mds'],
                                                                         h5['oids'],
                                                                         h5['conditions'],
                                                                         h5['starts'],
                                                                         h5['dts'],
                                                                         h5['columns'])):
                series = DataFrame(h5[str(oid)][:], columns=columns)
                trajs.append(FreeflightTrajectory(md, oid, start, cond, series, dt=dt))
            return trajs

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



#############################
# Other stuff for trajectories...
# self.stimulus = None       # Measurements directly related to the stimulus
#                            # We could infer this from column types, condition string...
#                            # But that is anyway a parameter of the analysis...
# We should also keep track on how the following were computed...
# But somewhere on a "data-analysis" level
# self.time_features = None  # The features derived from the (coords, stimulus) pairs (e.g. dtheta)
# self.summary_feats = None  # Summarizing features by collapsing time-series to 1D.
#                            # e.g. "mean dtheta" or "dtheta vs rotation rate correlation"
#
# We need to keep track on the preprocessing steps taken to arrive to a concrete trajectory...
#############################
