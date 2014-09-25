# coding=utf-8

#########################
#
# Ratio-based segmentation
#
#########################
#
# See also:
#  - calculate_nloops
#  - http://stackoverflow.com/questions/10475488/calculating-crossing-intercept-points-of-a-series-or-dataframe
#  - http://stackoverflow.com/questions/3843017/efficiently-detect-sign-changes-in-python
#  - crossings code in Etienne examples
#
#
# import numba
#
#
# @numba.autojit
# def crosses(x):
#     result = np.zeros_like(x, dtype=np.bool)
#     for i in xrange(len(x) - 1):
#         if x[i] > x[i + 1]:
#             result[i] = True
#     return result
#
#
# @numba.autojit
# def decreasing_in_a_row(x):
#     result = np.zeros_like(x, dtype=np.bool)
#     already_decreasing = False
#     for i in xrange(len(x) - 1):
#         if x[i] > x[i + 1]:
#             if already_decreasing:
#                 result[i] = True
#             already_decreasing = True
#         else:
#             already_decreasing = False
#     return result
#
# for i, traj in enumerate(trajs):
#     diw = decreasing_in_a_row(traj.series()['ratio'].values)
#     if np.sum(diw):
#         print i, traj.id_string(), np.sum(diw), np.where(diw)
#
#
# traj = df[(df['uuid'] == 'ad0377f0f95d11e38cd26c626d3a008a') &
#           (df['oid'] == 12430)].iloc[0].traj
#
#
# Look also at calc_unwrapped_ratio in curvature.py, although the bug, if there is one, is not there
#
#
# print np.sum(crosses(traj.df()['ratio'].values))
# print np.where(crosses(traj.df()['ratio'].values))
# traj.df().ratio.plot()
# import matplotlib.pyplot as plt
# plt.show()
#########################


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