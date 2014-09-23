#!/usr/bin/env python
import os.path
import sys
import numpy as np
import unittest
import collections
import tempfile
import random
import itertools

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')
import analysislib.combine
import analysislib.args
import analysislib.util
import nodelib.log

from flydra.analysis.save_as_flydra_hdf5 import save_as_flydra_hdf5
from ros_flydra.constants import IMPOSSIBLE_OBJ_ID

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
        traj_datatypes = []
        for i in "xyz":
            traj_datatypes.append( (i, np.float64) )
        traj_datatypes.append( ("obj_id", np.uint32) )
        traj_datatypes.append( ("framenumber", np.int64) )
        traj_start_datatypes = []
        traj_start_datatypes.append( ("obj_id", np.uint32) )
        for i in ("first_timestamp_secs", "first_timestamp_nsecs"):
            traj_start_datatypes.append( (i, np.uint64) )

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

if __name__=='__main__':
    unittest.main()

