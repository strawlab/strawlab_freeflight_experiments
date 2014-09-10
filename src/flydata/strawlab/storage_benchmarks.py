# coding=utf-8
"""Test and benchmark different alternatives to persistence of freeflight trajectories."""
import pandas as pd
import numpy as np
from flydata.example_analyses.dcn.dcn_data import load_lisa_dcn_experiments, DCN_UUIDs
from flydata.strawlab.metadata import FreeflightExperimentMetadata
from flydata.strawlab.trajectories import FreeflightTrajectory

#
# Desiderata / utopia for a storage backend (in no special order):
#   - interoperable (or at least not very esoteric)
#   - as opposed to pickle, do not break with changes in software
#   - simple
#   - compact: do not waste space
#   - enables parallel computation
#   - fast retrieval of vast amounts of numeric data
#   - allows for partial retrieval of data
#   - allows for quick check of already computed results (using e.g. Configurable)
#   - metadata not too far away
#
# Aspects not to ignore:
#   - were are we storing; it is not the same:
#     - a ramdisk (like /tmp in my machines)
#     - an SSD disk
#     - a spin disk
#     - networked storage backends (like /mnt/strawlab)
#   - data impedance (no problem with pickles vs hdf5 not able to store objects/unicode...)
#


###################
# The competitors
###################

class DiskTrajectoryStorage(object):

    def __init__(self):
        super(DiskTrajectoryStorage, self).__init__()

    def to_disk(self, trajs):
        pass

    def from_disk(self):
        pass

    def lazy_traj(self, trajid):
        pass


###################
# Assess correctness
###################

def same(traj1, traj2):
    raise NotImplementedError()


###############
# Benchmark datasets
###############


def artificial_trajectories(num_trajectories=10000, num_obs=1000):
    """Returns a list of "num_trajectories" trajectories, each with a few columns of "num_obs" length."""

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

    trajs = []
    for oid in xrange(num_trajectories):
        rng = np.random.RandomState(0)

        df = pd.DataFrame()
        df['x'] = rng.uniform(size=num_obs)
        df['y'] = rng.uniform(size=num_obs)
        df['z'] = rng.uniform(size=num_obs)
        df['rotation_rate'] = rng.uniform(size=num_obs)
        df['dtheta'] = rng.uniform(size=num_obs)

        traj = FreeflightTrajectory(md, oid, 100, 7777777887, condition='cool|1|2.5|blah.osg|23', series=df, dt=0.1)

        trajs.append(traj)

    return trajs


def dcn_dataset():
    """Returns a FreeflightExperiment list with all the data from Lisa's DCN experiments."""
    # N.B. at the moment we return experiments...
    return load_lisa_dcn_experiments(DCN_UUIDs)


def experiment_with_spurius_nonnumerics():
    """Returns a list with FreeflightExperiment corresponding to the uuid ae8425d4084911e4aafb6c626d3a008a.
    The pickle for this dataset contains "non-numeric" data in the dataframe
    (columns with things like uuid, condition...)
    If not treated correctly, this results in very slow numpy backends and impedance errors in hdf5 backends.
    """
    return load_lisa_dcn_experiments('ae8425d4084911e4aafb6c626d3a008a')


#####################################################
# if __name__ == '__main__':
#
#     from time import time
#     import os.path as op
#     start = time()
#     print 'Loading experiments...'
#     print 'There are %d experiments' % len(experiments)
#     conditions = set()
#     for exp in experiments:
#         print 'experiment=%s, genotype=%s' % (exp.uuid(), exp.md().genotype())
#         print 'Loading trajectories...'
#         trajs_cache = '/mnt/strawscience/--to-delete-storage-tests/%s.h5' % exp.uuid()
#         if not op.isfile(trajs_cache):
#             trajs = exp.trajectories()  # N.B. from network disk, unfair
#             FreeflightTrajectory.to_h5(trajs_cache, trajs)
#             # FreeflightTrajectory.to_npz(trajs_cache, trajs)
#         else:
#             trajs = FreeflightTrajectory.from_h5(trajs_cache, md=exp.md())
#             # trajs = FreeflightTrajectory.from_npz(trajs_cache, md=exp.md())
#         print 'There are %d conditions' % len(set(t.condition() for t in trajs))
#         conditions.update(t.condition() for t in trajs)
#         print '-' * 40
#     print 'Total number of conditions: %d' % len(conditions)
#     print 'Loading all data took %.2f seconds' % (time() - start)
#####################################################

#####################################################
#
# if __name__ == '__main__':
#
#     # N.B. /tmp is a ramdisk
#
#     print 'Generation took: %.2f' % (time.time() - start)
#
#     start = time.time()
#     with h5py.File('/tmp/test.h5') as h5:
#         trajs[0].to_h5(h5)
#         for traj in trajs[1:]:
#             traj.to_h5(h5, save_experiment_metadata=False, save_trajectory_metadata=False)
#     print 'Write took: %.2f' % (time.time() - start)
#
#     start = time.time()
#     with h5py.File('/tmp/test.h5', 'r') as h5:
#         exp_group = h5[u'uuid=FAKE_UUID_HERE_AND_THERE']
#         traj_ids = exp_group.keys()
#         for traj_id in traj_ids:
#             traj_group = exp_group[traj_id]
#             series_group = traj_group['series']
#             df = DataFrame({series_id: series_group[series_id].value for series_id in series_group})
#     print 'Reading from non-contiguous series took: %.2f' % (time.time() - start)
#
#     # Naive contiguous - upper bound of speed
#     X = np.random.uniform(size=(10000, 5000))
#     start = time.time()
#     with h5py.File('/tmp/test-fastest.h5') as h5:
#         h5['X'] = X
#     print 'Write contiguous: %.2f' % (time.time() - start)
#
#     start = time.time()
#     with h5py.File('/tmp/test-fastest.h5') as h5:
#         X = h5['X'].value
#         print X.shape
#     print 'Read contiguous: %.2f' % (time.time() - start)
#
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
#####################################################