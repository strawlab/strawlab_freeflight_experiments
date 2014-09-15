# coding=utf-8
import gzip
from itertools import product
import h5py
from pandas.io.pytables import HDFStore
from flydata.example_analyses.dcn.dcn_data import load_lisa_dcn_trajectories, ATO_TNTE, ATO_TNTin, \
    VT37804_TNTE, VT37804_TNTin, load_lisa_dcn_experiments
from time import time
from flydata.misc import ensure_dir
from flydata.strawlab.data_contracts import NoMissingValuesContract
from flydata.strawlab.metadata import FreeflightExperimentMetadata
from flydata.strawlab.trajectories import FreeflightTrajectory
import cPickle as pickle
import os.path as op
from oscail.common.config import Configurable


class PickleTrajectoryStorer(object):

    __MAGIC_NUMBER__ = 0

    def __init__(self, dest_dir, pkl_name='trajectories.pkl'):
        super(PickleTrajectoryStorer, self).__init__()
        self.dest_dir = dest_dir
        self.pkl_name = pkl_name

    def store(self, trajs):
        ensure_dir(self.dest_dir)
        with open(op.join(self.dest_dir, self.pkl_name), 'wb') as writer:
            pickle.dump(trajs, writer, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, traj_ids=None):
        with open(op.join(self.dest_dir, self.pkl_name), 'r') as reader:
            pickle.load(reader)

    def present_trajectories_ids(self):
        raise NotImplementedError()

    def present_conditions(self):
        raise NotImplementedError()

    def present_series(self):
        raise NotImplementedError()

    def present_features(self):
        raise NotImplementedError()


class NaiveHDF5TrajectoryStorer(object):

    __MAGIC_NUMBER__ = 0

    def __init__(self, dest_dir, hdf_name='trajectories.h5', compression='lzf', shuffle=False):
        super(NaiveHDF5TrajectoryStorer, self).__init__()
        self.dest_dir = dest_dir
        self.hdf_name = hdf_name
        self.compression = compression
        self.shuffle = shuffle

    def store(self, trajs):
        ensure_dir(self.dest_dir)
        with open(op.join(self.dest_dir, self.pkl_name), 'wb') as writer:
            pickle.dump(trajs, writer, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, traj_ids=None):
        with open(op.join(self.dest_dir, self.pkl_name), 'r') as reader:
            pickle.load(reader)

    def present_trajectories_ids(self):
        raise NotImplementedError()

    def present_conditions(self):
        raise NotImplementedError()

    def present_series(self):
        raise NotImplementedError()

    def present_features(self):
        raise NotImplementedError()


class FilterChain(object):
    pass


class ByNameFilter(Configurable):

    def __init__(self, series_to_keep):
        super(ByNameFilter, self).__init__(add_descriptors=False)
        self.series_to_keep = series_to_keep

    def fit(self, trajectories):
        return self

    def transform(self, trajectories):
        for traj in trajectories:
            traj.set_series(traj.series()[self.series_to_keep])  # Again, we might not want to do inplace the default
        return trajectories

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


def nanfill(df, series_names):
    for sname in series_names:
        ffill(df[sname].values)
        bfill(df[sname].values)


class NaNFiller(Configurable):

    def __init__(self,
                 series=('rotation_rate', 'trg_x', 'trg_y', 'trg_z', 'ratio'),
                 first_pass_method='ffill',
                 second_pass_method='bfill'):
        super(NaNFiller, self).__init__(add_descriptors=False)
        self.series_names = list(series)
        self.fpm = first_pass_method
        self.spm = second_pass_method

    def fit(self, trajectories):  # We might want to add Y to the API, ala sklearn
        return self

    def transform(self, trajs):
        for i, traj in enumerate(trajs):
            print i
            df = traj.series()
            # print df['ratio'].isnull().sum()
            # inplace not working + inplace should not be default
            # cannot make it work like this in 5 minutes...
            #   df.loc[self.series_names].fillna(method=self.fpm, inplace=True)
            #   if self.spm is not None:
            #       df.loc[self.series_names].fillna(method=self.spm, inplace=True)
            # so ugly for:
            for column in self.series_names:
                df[column] = df[column].fillna(method=self.fpm)  # N.B. inplace is damn slow in pandas 0.14
                                                                 # Should not be,
                                                                 # this requires only one pass on each col!
                if self.spm is not None:
                    df[column] = df[column].fillna(method=self.spm)
            # print df['ratio'].isnull().sum()
            # ANd this is awfully slow...
            traj.set_series(df)  # Depending on final trajectory semantics, this might be not necessary
            # TODO: Compare to fillna(method='ffill'), without selecting columns
        return trajs


completed_exps = ATO_TNTE + ATO_TNTin + VT37804_TNTE + VT37804_TNTin

CACHE_DIR = op.join(op.expanduser('~'), 'data-analysis', 'strawlab', 'dcns', '20140909', 'original')

#
# CACHE THE TRAJECTORIES AND THE METADATA
#
# for uuid in completed_exps:
#     exp = load_lisa_dcn_experiments(uuid)[0]
#     exp.sfff().mirror_to(op.join(CACHE_DIR, uuid))
# exit(22)
#

#
# REPICKLE STUFF
# N.B. we should also change the dictionary in data.json to normalize condition strings also there
# for uuid in completed_exps:
#     print uuid
#     exp = load_lisa_dcn_experiments(uuid, cache_root_dir=CACHE_DIR)[0]
#     exp.sfff().repickle()
# exit(33)
#


def dcn_conflict_select_interesting_columns(
    rotation_rate=True,  # speed of the stimulus rotation
    trg_x=True,          # x towards which the rotation stimulus is pushing the fly to
    trg_y=True,          # y towards which the rotation stimulus is pushing the fly to
    trg_z=True,          # z towards which the rotation stimulus is pushing the fly to
    cyl_x=False,
    cyl_y=False,
    cyl_r=False,
    ratio=True,          # [0,1] infinity loop position from the center of the infinity figure
    v_offset_rate=False,
    phase=False,
    model_x=False,
    model_y=False,
    model_z=False,
    model_filename=False,
    condition=False,
    lock_object=False,
    t_sec=False,
    t_nsec=False,
    flydra_data_file=False,
    exp_uuid=False,
    # Response information
    x=True,
    y=True,
    z=True,
    tns=False,
    vx=True,
    vy=True,
    vz=True,
    velocity=True,
    ax=True,
    ay=True,
    az=True,
    theta=True,
    dtheta=True,
    radius=True,
    omega=True,
    rcurve=True,
    framenumber=False,
):
    return [column for column, useful in sorted(locals().items()) if useful]

INTERESTING_SERIES = dcn_conflict_select_interesting_columns()
STIMULI_SERIES = ('rotation_rate', 'trg_x', 'trg_y', 'trg_z', 'ratio')

# Load the trajectories
start = time()
trajs = load_lisa_dcn_trajectories(uuids=completed_exps[1], cache_root_dir=CACHE_DIR)
# trajs = load_lisa_dcn_trajectories(uuids=completed_exps, cache_root_dir=CACHE_DIR)
# with open(op.join(CACHE_DIR, 'cached.pkl')) as reader:
#     trajs = pickle.load(reader)
print 'Loading took %.2f seconds' % (time() - start)

# Apply filters

# Filter-out uninteresting and non-numeric series
trajs = ByNameFilter(series_to_keep=INTERESTING_SERIES).fit(trajs).transform(trajs)  # Is this correct for trg_x
                                                                                     # and the like?
# Fill missing values in stimuli data
start = time()
# trajs = NaNFiller(series=STIMULI_SERIES).fit(trajs).transform(trajs)
for traj in trajs:
    nanfill(traj.series(), STIMULI_SERIES)
print 'Pandas took %.2f seconds' % (time() - start)


# with open(op.join(CACHE_DIR, 'cached.pkl'), 'wb') as writer:
#     pickle.dump(trajs, writer, protocol=pickle.HIGHEST_PROTOCOL)

# Check contracts
# no_missings_please = NoMissingValuesContract(columns=STIMULI_SERIES)
no_missings_please = NoMissingValuesContract(columns=trajs[0].series().columns)
for agree, rows in no_missings_please.check(trajs):
    if not agree:
        print rows
        raise Exception('There are missings...')

exit(22)

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


for uuid in completed_exps[2:3]:
    print uuid

    # Load using
    trajs = load_lisa_dcn_trajectories(uuids=uuid, cache_root_dir=CACHE_DIR)

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

    ################################
    # Pytables
    ################################

    import warnings
    import tables
    warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)

    complevels = range(5, 6)
    shuffles = shuffles = [True, False]
    # zlib, lzo, bzip2, blosc, blosc:blosclz, blosc:lz4, blosc:lz4hc, blosc:snappy, blosc:zlib
    # comps = ('blosc', 'lzo', 'blosc:snappy', 'blosc:lz4', 'blosc:blosclz', 'blosc:zlib')
    comps = ('blosc', 'lzo', 'blosc:snappy', 'blosc:lz4', 'blosc:blosclz', 'blosc:zlib')
    for complevel, shuffle, comp in product(complevels, shuffles, comps):
        fn = '/home/santi/tables_%s_complevel=%d_shuffle=%r.h5' % (comp, complevel, shuffle)
        with tables.open_file(fn, mode='a') as h5:
            filters = tables.Filters(complib=comp, complevel=complevel, shuffle=shuffle)
            for traj in trajs:
                df = traj.series()._get_numeric_data()
                ds = h5.createCArray(h5.root,
                                     traj.id_string(),
                                     tables.Atom.from_dtype(df.values.dtype),
                                     df.values.shape,
                                     filters=filters)
                ds[:] = df.values



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


print 'Loading took %.2f seconds' % (time() - start)

# print 'Loading %d trajectories took %.2f seconds' % (len(trajs), time() - start)

# import cPickle as pickle
# pickle the trajectories...
# FreeflightTrajectory.to_npz('/home/santi/dcn-trajs.h5', trajs,  compress_mds=False)
# FreeflightTrajectory.to_h5('/home/santi/dcn-trajs.h5', trajs, compress_mds=False)
# FreeflightTrajectory.to_h5_identifiable('/home/santi/dcn-trajs-identifiable.h5')

exit(33)

for trajnum, traj in enumerate(trajs):

    print 'Trajectory %d: %s (%s)' % (trajnum, traj.id_string(), traj.condition())

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

    #
    # Now the dataframes come loaded with many (un)interesting columns:

    # Pad missing values in trg: best directions are 1-forward 2-backward
    # Why are there so many missing values in these columns, is it normal?
    df = traj.series()
    df['trg_x'].fillna(method='ffill', inplace=True)
    df['trg_x'].fillna(method='bfill', inplace=True)
    df['trg_y'].fillna(method='ffill', inplace=True)
    df['trg_y'].fillna(method='bfill', inplace=True)

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

# print df.model_x, df.model_y, df.model_z

#
# TODO: small application that generates a "reduced HTML" view of a "project"
#       could be as simple as copy/paste the results webserver initial page
#
# TODO: for max, allow to change the metadata programmatically
# (could be the opportunity to add file/db sync to MD)
#