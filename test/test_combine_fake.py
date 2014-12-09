#!/usr/bin/env python
import os.path
import sys
import numpy as np
import unittest
import collections
import tempfile
import random
import itertools
from pandas.core.groupby import GroupBy
from autodata.files import FileModel

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')
import analysislib.combine
import analysislib.args
import analysislib.util
import nodelib.log

from flydra.analysis.save_as_flydra_hdf5 import save_as_flydra_hdf5
from ros_flydra.constants import IMPOSSIBLE_OBJ_ID


# --- test data generation utils

def simple_flydra_datatypes():
    """Returns a 2-tuple (traj_datatypes, traj_start_datatypes).
    Useful to store trajectories ala flydra.
    """

    traj_datatypes = [
        ("x", np.float64),
        ("y", np.float64),
        ("z", np.float64),
        ("obj_id", np.uint32),
        ("framenumber", np.int64),
    ]

    traj_start_datatypes = [
        ("obj_id", np.uint32),
        ("first_timestamp_secs", np.uint64),
        ("first_timestamp_nsecs", np.uint64),
    ]

    return traj_datatypes, traj_start_datatypes


def combine2h5csv(combine,
                  trim_trajs_to=10,
                  columns_for_csv=('rotation_rate', 'ratio'),
                  tempdir=None,
                  file_prefix='data',
                  fps=100):
    """
    Generates ".simple.flydra.h5" and ".csv" files from combine objects.

    This could be called too "uncombine".

    Heavily inspired by "TestCombineFake2._create_h5".
    """
    # There is danger of creating unrealistic csv/h5 files.
    # That is why, even if writing them directly is not a big deal,
    # we better pass by:
    #    - nodelib.CSVLogger to write the CSV
    #    - flydra save_as_flydra_hdf5 to write the H5 file

    # dest files
    if tempdir is None:
        tempdir = tempfile.mkdtemp()
    h5_fname = os.path.join(tempdir, "%s.simple_flydra.h5" % file_prefix)
    csv_fname = os.path.join(tempdir, "%s.csv" % file_prefix)

    # fake csv
    log_state = columns_for_csv
    log = nodelib.log.CsvLogger(fname=csv_fname,
                                wait=False, debug=False, warn=False,
                                state=log_state)
    # N.B. we are not populating the "flydra_data_file" field

    # fake h5 file
    traj_datatypes, traj_start_datatypes = simple_flydra_datatypes()
    traj_data = {k[0]: [] for k in traj_datatypes}
    traj_starts_data = {k[0]: [] for k in traj_start_datatypes}

    # populate and write
    results, dt = combine.get_results()
    for cond, condtrials in results.iteritems():
        for uuid, (x0, y0, obj_id, framenumber0, time0), df in zip(condtrials['uuids'],
                                                                   condtrials['start_obj_ids'],
                                                                   condtrials['df']):
            # select only the first n observations
            if trim_trajs_to is not None:
                df = df.head(n=trim_trajs_to)

            # prepare the csv logger
            log.condition = cond
            log.lock_object = obj_id
            log.framenumber = framenumber0
            log._exp_uuid = uuid  # dirty
            # write the csv rows for this trial
            for _, row in df.iterrows():
                for col in columns_for_csv:
                    setattr(log, col, row[col])
                log.update()

            # accummulate the data to write the h5 at the end
            for col in 'xyz':
                traj_data[col].extend(df[col].values)
            traj_data['framenumber'].extend(df.index.values)  # this could be easily generalized to other index types
            traj_data['obj_id'].extend(itertools.repeat(obj_id, len(df)))
            traj_starts_data["obj_id"].append(obj_id)
            traj_starts_data["first_timestamp_secs"].append(log.last_tsecs)  # N.B. not time0
            traj_starts_data["first_timestamp_nsecs"].append(log.last_tnsecs)

    # flatten all the data lists into numpy arrays

    npts = len(traj_data['obj_id'])
    traj_arr = np.zeros(npts, dtype=traj_datatypes)
    for k in traj_data:
        traj_arr[k] = traj_data[k]

    npts = len(traj_starts_data['obj_id'])
    traj_start_arr = np.zeros(npts, dtype=traj_start_datatypes)
    for k in traj_starts_data:
        traj_start_arr[k] = traj_starts_data[k]

    # save to "simple flydra h5"
    # N.B. this compresses using gzip+9; that is slow, is all saved like this?
    # N.B. this does not save "experiment_info", a table with the uuid, which seems optional
    save_as_flydra_hdf5(h5_fname,
                        {"trajectories": traj_arr, "trajectory_start_times": traj_start_arr},
                        "US/Pacific",
                        fps)
    log.close()

    return csv_fname, h5_fname

# alias
uncombine = combine2h5csv


# --- tests for some combine functionality

class TestCombineFake(unittest.TestCase):

    def setUp(self):
        self._tdir = tempfile.mkdtemp()
        self.combine = analysislib.combine._CombineFakeInfinity(nconditions=3)
        parser,self.args = analysislib.args.get_default_args(
                outdir='/tmp/',
                lenfilt=0,
                show='--show' in sys.argv
        )
        self.combine.add_from_args(self.args)
        self.df0 = self.combine.get_one_result(1)[0]

    def _check_oid1(self, df,dt,x0,y0,obj_id,framenumber0,time0):
        self.assertEqual(dt, self.combine._dt)
        self.assertEqual(x0, 0.0)
        self.assertEqual(y0, 0.0)
        self.assertEqual(obj_id, 1)
        self.assertEqual(framenumber0, 1)

    def test_load(self):
        #be default we have no filter, so we get 300 trials
        self.assertEqual(self.combine.min_num_frames, 0)
        self.assertEqual(len(self.combine.get_conditions()), 3)
        for c in self.combine.get_conditions():
            self.assertEqual(self.combine.get_num_trials(c), 100)
            self.assertEqual(self.combine.get_num_analysed(c), 100)
            self.assertEqual(self.combine.get_num_skipped(c), 0)
        self.assertEqual(self.combine.get_total_trials(), 300)

        self.assertRaises(ValueError, self.combine.get_one_result, 0)

        df,dt,(x0,y0,obj_id,framenumber0,time0) = self.combine.get_one_result(1)
        self._check_oid1(df,dt,x0,y0,obj_id,framenumber0,time0)

        self.assertEqual(time0, self.combine._t0 + self.combine._dt)

    def test_custom_filter(self):
        # FIXME: this tests some stuff that should not be used, as create holes in trajectories
        #        just remove custom_filt
        c1 = analysislib.combine._CombineFakeInfinity(nconditions=1,ntrials=1)
        c1.add_custom_filter("df[(df['ratio']>0.2)&(df['ratio']<0.8)]", 2.0)
        c1.add_from_args(self.args)

        #the filtered dataframe should be shorter than the original one
        df,dt,(x0,y0,obj_id,framenumber0,time0) = c1.get_one_result(1)
        self.assertLess(len(df), len(self.df0))

        c1 = analysislib.combine._CombineFakeInfinity(nconditions=1,ntrials=1)
        c1.add_custom_filter("df[(df['velocity']>1.22)]", 3.0)
        c1.add_from_args(self.args)

        #the filtered dataframe should be shorter than the original one
        df,dt,(x0,y0,obj_id,framenumber0,time0) = c1.get_one_result(1)
        self.assertLess(len(df), len(self.df0))

        c1 = analysislib.combine._CombineFakeInfinity(nconditions=1,ntrials=1)
        c1.add_custom_filter("df[(df['velocity']>9999)]", 1.0)
        c1.add_from_args(self.args)

        #no data left after filtering
        self.assertRaises(ValueError, c1.get_one_result, 1)

    def test_load_from_args(self):
        c1 = analysislib.combine._CombineFakeInfinity(nconditions=1,ntrials=1)
        parser,args = analysislib.args.get_default_args(
                outdir='/tmp/',
                show='--show' in sys.argv,
                customfilt="df[(df['ratio']>0.2)&(df['ratio']<0.8)]",
                customfilt_len=1.0,
                lenfilt=1
        )
        c1.add_from_args(args)

        #the filtered dataframe should be shorter than the original one
        df,dt,(x0,y0,obj_id,framenumber0,time0) = c1.get_one_result(1)
        self.assertLess(len(df), len(self.df0))

    def test_export(self):
        d = os.path.join(self._tdir, "export")
        dests = analysislib.combine.write_result_dataframe(d,self.df0, self.combine._index)
        self.assertEqual(os.path.basename(dests),"export_framenumber.{csv,df,mat}")
        dests = analysislib.combine.write_result_dataframe(d,self.df0, self.combine._index,to_mat=False)
        self.assertEqual(os.path.basename(dests),"export_framenumber.{csv,df}")

class TestCombineFake2(unittest.TestCase):

    def setUp(self):
        self._tdir = tempfile.mkdtemp()

    def _create_h5(self, npts, noids, csv_rate=1, framenumber0=1):

        def _repeat_or_divide_iter(it, n):
            #if n is > 1 then return each element n times, else if it is less
            #than one return in total n% of the total elements
            for i in it:
                if n < 1:
                    if random.random() <= n:
                        yield i
                    else:
                        continue
                else:
                    for _ in range(int(n)):
                        yield i

        h5_fname = os.path.join(self._tdir,"data.simple_flydra.h5")
        csv_fname = os.path.join(self._tdir,"data.csv")

        #create the numpy datatypes
        traj_datatypes, traj_start_datatypes = simple_flydra_datatypes()

        #we need to accumulate trajectory data from all flies (obj_ids)
        traj_data = {k[0]:[] for k in traj_datatypes}
        traj_starts_data = {k[0]:[] for k in traj_start_datatypes}

        #create a fake csv
        log_state = ("rotation_rate","ratio")
        log = nodelib.log.CsvLogger(fname=csv_fname,
                                    wait=False, debug=False, warn=False,
                                    state=log_state)
        log.condition = 'test'
        log.lock_object = IMPOSSIBLE_OBJ_ID
        log.framenumber = 0

        FPS = 100

        frame0 = framenumber0

        for oid in range(1,noids+1):

            log.framenumber = frame0
            log.lock_object = IMPOSSIBLE_OBJ_ID
            log.update()

            frame0 += 1

            df = analysislib.combine._CombineFakeInfinity.get_fake_infinity(
                                n_infinity=1,
                                random_stddev=0,
                                x_offset=0,y_offset=0,
                                frame0=frame0,
                                nan_pct=0,
                                latency=0,dt=1.0/FPS,
                                npts=npts)

            #wait this many frames before 'locking on', i.e writing the csv
            #file
            lock_on_delay = int(random.random() * 10)
            just_locked_on = None

            #because the csv shares the framerate with the dataframe we might
            #need to duplicate or repeat those values
            for i,(ix,row) in enumerate(_repeat_or_divide_iter(df.iterrows(), csv_rate)):
                if i >= lock_on_delay:
                    #we are locked on
                    if just_locked_on is None:
                        just_locked_on = True

                    for l in log_state:
                        setattr(log,l,row[l])
                    log.framenumber = ix
                    log.lock_object = oid
                    log.update()

                    if just_locked_on is True:
                        #first time through this loop
                        traj_starts_data["obj_id"].append(oid)
                        traj_starts_data["first_timestamp_secs"].append(log.last_tsecs)
                        traj_starts_data["first_timestamp_nsecs"].append(log.last_tnsecs)
                        just_locked_on = False

            #extract the trajectory from the dataframe
            for i in 'xyz':
                traj_data[i].extend( df[i].values )
            traj_data['framenumber'].extend(df.index.values)
            traj_data['obj_id'].extend(itertools.repeat(oid,len(df)))

            frame0 += npts

        #create the fake simple_flydra.h5 from all data for all flies
        npts = len(traj_data['obj_id'])
        traj_arr = np.zeros( npts, dtype=traj_datatypes )
        for k in traj_data:
            traj_arr[k] = traj_data[k]

        npts = len(traj_starts_data['obj_id'])
        traj_start_arr = np.zeros( npts, dtype=traj_start_datatypes )
        for k in traj_starts_data:
            traj_start_arr[k] = traj_starts_data[k]

        data = {"trajectories":traj_arr,
                "trajectory_start_times":traj_start_arr}

        save_as_flydra_hdf5(h5_fname, data, "US/Pacific", FPS)
        log.close()

        return h5_fname, csv_fname

    def test_create_h5(self):
        h5_fname, _ = self._create_h5(500,1,1)
        combine = analysislib.combine.CombineH5()
        combine.add_h5_file(h5_fname)
        df,dt,(x0,y0,obj_id,framenumber0,time0) = combine.get_one_result(1)

        self.assertEqual(len(df),500)
        self.assertEqual(obj_id,1)
        self.assertEqual(framenumber0,2)

    def test_combine_h5_csv(self):

        parser,args = analysislib.args.get_default_args(
                        outdir=self._tdir,
                        rfilt='none',zfilt='none')

        for index,csv_rate in itertools.product(('none','framenumber','time','time+1L'),(0.5,2.0)):
            h5_fname, csv_fname = self._create_h5(200,1,csv_rate=csv_rate,framenumber0=14)
            cn = analysislib.util.get_combiner_for_csv(csv_fname)
            cn.calc_angular_stats = False
            cn.calc_turn_stats = False
            cn.calc_linear_stats = False
            cn.set_index(index)

            cn.add_csv_and_h5_file(csv_fname, h5_fname, args)
            dfn,dt,(x0,y0,obj_id,framenumber0,time0) = cn.get_one_result(1)

            self.assertEqual(obj_id,1)


#
# Test:
#    - uuids storage
#    - pickle
#    - contiguous-condition trajectories
#
# Need to install:
#    - python-whatami
#

if __name__ == '__main__':

    from analysislib.util import get_combiner_for_uuid, get_combiner_for_csv
    import os.path as op
    import glob

    # Use FileModel here would make this somehow easier?
    MAX_UUID = '6d7142fc643d11e4be3d60a44c2451e5'
    DATA_ROOT = op.join(op.expanduser('~'), 'data-analysis', 'strawlab', 'combine', 'max-long', MAX_UUID)
    csv = glob.glob(op.join(DATA_ROOT, '*.csv'))[0]
    h5 = glob.glob(op.join(DATA_ROOT, '*.simple*'))[0]

    parser, args = analysislib.args.get_default_args(rfilt='none',
                                                     zfilt='none')
    combine = get_combiner_for_csv(csv)
    # TODO: calc_turn_stats and friends should be configurable by CL or documented
    # calc_turn_stats is time consuming, tell Max to disable (+look at memory +look at SMB/NFS bottlenecks with time)
    combine.calc_turn_stats = False
    combine.calc_linear_stats = False
    combine.calc_angular_stats = False
    combine.add_csv_and_h5_file(csv_fname=csv,
                                h5_file=h5,
                                args=args)

    # Adding from UUID
    # combine = get_combiner_for_uuid(MAX_UUID)
    # combine.add_from_uuid(MAX_UUID, rfilt='none', zfilt='none')  # improve this...

    combine2h5csv(combine,
                  tempdir=op.expanduser('~'),
                  columns_for_csv=('stim_x',))

    # results, dt = combine.get_results()
    # for cond, condtrials in results.iteritems():
    #     uuids = condtrials['uuids']
    #     meta = condtrials['start_obj_ids']
    #     dfs = condtrials['df']
    #     for uuid, (x0, y0, obj_id, framenumber0, time0), df in zip(uuids, meta, dfs):
    #         print cond, uuid, obj_id, framenumber0, len(df) * dt


if __name__=='__main__':
    unittest.main()

