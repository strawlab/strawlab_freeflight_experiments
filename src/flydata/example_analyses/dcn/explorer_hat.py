# coding=utf-8
from time import time
import cPickle as pickle
import os.path as op
import numpy as np

from flydata.example_analyses.dcn.dcn_data import load_lisa_dcn_trajectories, dcn_conflict_select_columns, \
    DCN_COMPLETED_EXPERIMENTS, load_lisa_dcn_experiments, DCN_CONFLICT_CONDITION, DCN_ROTATION_CONDITION
from flydata.strawlab.contracts import NoMissingValuesContract, NoHolesContract, AllNumericContract, check_contracts
from flydata.strawlab.experiments import FreeflightExperiment
from flydata.strawlab.trajectories import FreeflightTrajectory
from flydata.strawlab.transformers import MissingImputer, ColumnsSelector, NumericEnforcer, RowsWithMissingRemover


# A (local) directory in which we will store data and results
CACHE_DIR = op.join(op.expanduser('~'), 'data-analysis', 'strawlab', 'dcns', '20140909', 'original')

# What we are going to keep from the combined trajectories
INTERESTING_SERIES = dcn_conflict_select_columns()

# These are the stimuli series (we will need to impute missing values for these columns)
STIMULI_SERIES = ('rotation_rate', 'trg_x', 'trg_y', 'trg_z', 'ratio')


def first_read(uuids=DCN_COMPLETED_EXPERIMENTS,
               cache_dir=CACHE_DIR,
               mirror_locally=False,
               force=False):

    # Local mirror of combined trajectories and metadata
    if mirror_locally:
        for exp in load_lisa_dcn_experiments(uuids):
            exp.sfff().mirror_to(op.join(CACHE_DIR, exp.uuid()))

    cache_file = op.join(cache_dir, 'initial_data.pkl')

    if not op.isfile(cache_file) or force:

        # Load the trajectories from the local cache
        start = time()
        trajs = load_lisa_dcn_trajectories(uuids=uuids, cache_root_dir=CACHE_DIR)
        print 'Loading took %.2f seconds' % (time() - start)

        # Apply transformers
        TRANSFORMERS = (
            # Filter-out uninteresting series
            ColumnsSelector(series_to_keep=INTERESTING_SERIES),
            # Make all series numeric
            NumericEnforcer(),
            # Fill missing values in stimuli data (make sure it makes sense for all the series)
            MissingImputer(columns=STIMULI_SERIES, faster_if_available=True),
            # Filter-out trajectories with missing values in the responses (we do not know yet why that happened...)
            RowsWithMissingRemover(log_removed=True)
        )

        start = time()
        for transformer in TRANSFORMERS:
            print transformer.configuration().id()
            trajs = transformer.fit_transform(trajs)
        print 'Transformations took %.2f seconds, there are %d trajectories left' % (time() - start, len(trajs))

        # Check contracts
        CONTRACTS = (
            NoMissingValuesContract(columns=INTERESTING_SERIES),  # No missing values, please
            NoHolesContract(),                                    # No holes in time series, please
            AllNumericContract(),                                 # No "object" columns, please
        )

        start = time()
        checked = check_contracts(trajs, CONTRACTS)
        print 'Checked:\n\t%s' % '\n\t'.join(checked)
        print 'Check contracts took %.2f seconds' % (time() - start)

        # Save to our extraordinary data format (a pickle)
        with open(cache_file, 'wb') as writer:
            pickle.dump(trajs, writer, protocol=pickle.HIGHEST_PROTOCOL)

    with open(cache_file) as reader:
        return pickle.load(reader)

    # TODO: save also data path (transformers + checkers)

print 'Loading all trajectories, after initial transformations and sanity checks'
start = time()
trajs = first_read()
print 'Read %d trajectories in %.2f seconds' % (len(trajs), time() - start)


# We can make a pandas dataframe containing the trajectories
df = FreeflightTrajectory.to_pandas(trajs)

# Let's group trajectories by conflict condition (yes, we could also use groupby)
trajs_on_conflict = df[df['condition'] == DCN_CONFLICT_CONDITION]
trajs_on_rotation = df[df['condition'] == DCN_ROTATION_CONDITION]
assert len(df) == len(trajs_on_conflict) + len(trajs_on_rotation)

# Let's group trajectories by condition and genotype...
print df.groupby(by=('condition', 'genotype'))['traj'].count()
# ...or by condition and experiment...
print df.groupby(by=('condition', 'genotype'))['traj'].count()
# ...or by night/day genotype...
df['night'] = df['traj'].apply(lambda x: x.is_between_hours())
print df.groupby(by=('night', 'genotype'))['traj'].count()

#
# OK, those counts can actually be misleading, how many trajectories per hour could be better
# e.g. if there are 3 hours of day and 9 of night...
#
# In these metadatas we usually do not have a "experiment_stop" time.
# Instead we can approximate the duration of an experiment as the difference
# between the starting times of the first and the last trajectories
#
# Let's do it:
#
# roughly_exp_durations = df.groupby('uuid')['start'].max() - df.groupby('uuid')['start'].min()
# roughly_exp_durations = roughly_exp_durations.apply(lambda x: x / np.timedelta64(1, 's'))
# roughly_daylight_duration = blah
# roughly_night_duration = bleh
# df['exp_duration'] = df['uuid'].apply(lambda uuid: roughly_exp_durations[uuid])  # Lots of DRY
# print df['exp_duration']
#

#
# See also:
#  - calculate_nloops
#  - http://stackoverflow.com/questions/10475488/calculating-crossing-intercept-points-of-a-series-or-dataframe
#  - http://stackoverflow.com/questions/3843017/efficiently-detect-sign-changes-in-python
#  - crossings code in Etienne examples
#

import numba


@numba.autojit
def crosses(x):
    result = np.zeros_like(x, dtype=np.bool)
    for i in xrange(len(x) - 1):
        if x[i] > x[i + 1]:
            result[i] = True
    return result


@numba.autojit
def decreasing_in_a_row(x):
    result = np.zeros_like(x, dtype=np.bool)
    already_decreasing = False
    for i in xrange(len(x) - 1):
        if x[i] > x[i + 1]:
            if already_decreasing:
                result[i] = True
            already_decreasing = True
        else:
            already_decreasing = False
    return result

for i, traj in enumerate(trajs):
    diw = decreasing_in_a_row(traj.series()['ratio'].values)
    if np.sum(diw):
        print i, traj.id_string(), np.sum(diw), np.where(diw)


traj = df[(df['uuid'] == 'ad0377f0f95d11e38cd26c626d3a008a') &
          (df['oid'] == 12430)].iloc[0].traj

#
# Look also at calc_unwrapped_ratio in curvature.py, although the bug, if there is one, is not there
#

print np.sum(crosses(traj.df()['ratio'].values))
print np.where(crosses(traj.df()['ratio'].values))
traj.df().ratio.plot()
import matplotlib.pyplot as plt
plt.show()

#############################################################
# OLD USELES STUFF TO REVIEW
#############################################################
#
# Save to an hdf5 file
# with h5py.File('/home/santi/dcntrajs.h5', 'w') as h5:
#     for traj in trajs:
#         try:
#             uuidg = h5['expid=%s' % traj.uuid]
#         except:
#             uuidg = h5.require_group('expid=%s' % traj.uuid)
#             uuidg.attrs['md'] = traj.md().to_json_string()
#         trajg = uuidg.create_group(traj.id_string())
#         trajg.attrs['oid'] = traj.oid()
#         trajg.attrs['start_frame'] = traj.start_frame()
#         trajg.attrs['start_time'] = traj.start_time()
#         trajg.attrs['condition'] = traj.condition()
#         trajg.attrs['dt'] = traj.dt()
#         trajg.create_dataset('series', data=traj.series(), compression='lzf', shuffle=False)
#
# start = time()
# trajs = []
# with h5py.File('/home/santi/dcntrajs.h5', 'r') as h5:
#     for _, uuidg in h5.iteritems():
#         md = FreeflightExperimentMetadata.from_json_string(uuidg.attrs['md'])
#         for trajg in uuidg.values():
#             attrs = trajg.attrs
#             trajs.append(
#                 FreeflightTrajectory(md,
#                                      attrs['oid'],
#                                      attrs['start_frame'],
#                                      attrs['start_time'],
#                                      attrs['condition'],
#                                      trajg['series'][:],
#                                      dt=attrs['dt']))
#                 # FreeflightTrajectory(md,
#                 #                      None,
#                 #                      None,
#                 #                      None,
#                 #                      None,
#                 #                      trajg['series'][:],
#                 #                      dt=None))  # Attribute access in HDF5 is SLOW, maybe save again as arrays
# print '%d: %.2f' % (len(trajs), time() - start)
# exit(33)
#
#
#
################################
# Pickle
################################
# import cPickle as pickle
#
# with open('/home/santi/lisadcns/%s/data.pkl' % uuid, 'wb') as writer:
#     pickle.dump(trajs, writer, pickle.HIGHEST_PROTOCOL)
#
# with open('/home/santi/lisadcns/%s/data.pkl' % uuid, 'r') as reader:
#     pickle.load(reader)
#
# with gzip.open('/home/santi/lisadcns/%s/data.pkl.gz' % uuid, 'wb') as writer:
#     pickle.dump(trajs, writer, pickle.HIGHEST_PROTOCOL)
#
# with gzip.open('/home/santi/lisadcns/%s/data.pkl.gz' % uuid, 'r') as reader:
#     pickle.load(reader)
################################
#
################################
# Pytables
################################
#
# import warnings
# import tables
# warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)
#
# complevels = range(5, 6)
# shuffles = shuffles = [True, False]
# # zlib, lzo, bzip2, blosc, blosc:blosclz, blosc:lz4, blosc:lz4hc, blosc:snappy, blosc:zlib
# # comps = ('blosc', 'lzo', 'blosc:snappy', 'blosc:lz4', 'blosc:blosclz', 'blosc:zlib')
# comps = ('blosc', 'lzo', 'blosc:snappy', 'blosc:lz4', 'blosc:blosclz', 'blosc:zlib')
# for complevel, shuffle, comp in product(complevels, shuffles, comps):
#     fn = '/home/santi/tables_%s_complevel=%d_shuffle=%r.h5' % (comp, complevel, shuffle)
#     with tables.open_file(fn, mode='a') as h5:
#         filters = tables.Filters(complib=comp, complevel=complevel, shuffle=shuffle)
#         for traj in trajs:
#             df = traj.series()._get_numeric_data()
#             ds = h5.createCArray(h5.root,
#                                  traj.id_string(),
#                                  tables.Atom.from_dtype(df.values.dtype),
#                                  df.values.shape,
#                                  filters=filters)
#             ds[:] = df.values
#
################################
# Pandas HDFStore
################################
#
# import warnings
# import tables
# warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)
#
# store = HDFStore('/home/santi/tables_bzip2.h5',
#                  mode='w',
#                  complevel=9,
#                  complib='zlib',
#                  shuffle=False,
#                  fletcher32=False)
#
# for i, traj in enumerate(trajs):
#     df = traj.series()._get_numeric_data()
#     # store.put('uuid=%s/%s' % (uuid, traj.id_string()), traj.series())
#     store.put('uuid=%s/%s' % (uuid, str(i)), df)
#
################################
#
# with h5py.File('/home/santi/h5py3.h5', 'w') as h5:
#     for traj in trajs:
#         df = traj.series()
#         # cols = [col for col, dtype in df.dtypes.iteritems() if dtype != np.object]  # Not the proper check
#         # print cols
#         # df = df[cols]
#         df = df._get_numeric_data()
#         # h5.create_dataset(traj.id_string(), data=df, compression='lzf', shuffle=True)
#         # h5.create_dataset(traj.id_string(), data=df, compression='lzf', shuffle=False)
#         h5.create_dataset(traj.id_string(), data=df, compression=None)
#
# print 'Loading took %.2f seconds' % (time() - start)
#
# print 'Loading %d trajectories took %.2f seconds' % (len(trajs), time() - start)
#
# import cPickle as pickle
# pickle the trajectories...
# FreeflightTrajectory.to_npz('/home/santi/dcn-trajs.h5', trajs,  compress_mds=False)
# FreeflightTrajectory.to_h5('/home/santi/dcn-trajs.h5', trajs, compress_mds=False)
# FreeflightTrajectory.to_h5_identifiable('/home/santi/dcn-trajs-identifiable.h5')
#
# Basicmost plot to see what is going on...
# import matplotlib.pyplot as plt
# _, (ax1, ax2) = plt.subplots(2, sharex=True)
# traj.series()[['x', 'trg_x']].plot(ax=ax1)
# traj.series()[['y', 'trg_y']].plot(ax=ax2)
# plt.suptitle('%d: %s' % (i, traj.id_string()))
# plt.tight_layout()
# plt.savefig('/home/santi/%d-%s.png' % (i, traj.id_string().replace('#', '__')))
# plt.close()
#
# Now the dataframes come loaded with many (un)interesting columns:
#
# Pad missing values in trg: best directions are 1-forward 2-backward
#
#
# Find constant segments
# (might mark better boundaries for valid trials than zfilt/rfilt/lenfilt)
# (but of course fillna would have already created some artificial ones)
#
# If we look for just equality, we can of course use groupby:
#   http://stackoverflow.com/questions/14358567/finding-consecutive-segments-in-a-pandas-data-frame
# If we look for, for example, slow changing segments
#   we could use diff or the like
#   look at the code I wrote for Etienne a while ago
#
#
################################################################################
#
# TODO: small application that generates a "reduced HTML" view of a "project"
#       could be as simple as copy/paste the results webserver initial page
#
# TODO: for max, allow to change the metadata programmatically
# (could be the opportunity to add file/db sync to MD)
#
################################################################################