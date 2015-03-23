#!/usr/bin/env python
import os.path
import os.path as op
import unittest
import tempfile
import random
import itertools
import sys

import numpy as np
import pandas as pd

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')

from analysislib.util import get_combiner_for_csv
from analysislib.util import get_combiner_for_uuid
import analysislib.combine
import analysislib.args
import analysislib.util
import nodelib.log

from ros_flydra.constants import IMPOSSIBLE_OBJ_ID
from flydra.analysis.save_as_flydra_hdf5 import save_as_flydra_hdf5
from strawlab_freeflight_experiments.conditions import Condition

from test_combine_api import _quiet


# --- test data generation utils


def select_h5csv_oids(h5_file='/mnt/strawscience/data/csv-test-case-deleteme/katja-exp/'
                              '20150131_174031.simple_flydra.h5',
                      csv_file='/mnt/strawscience/data/csv-test-case-deleteme/katja-exp/'
                               'overlap-e3ccf472a96711e48156bcee7bdac428-5537-595668.csv',
                      h5_dest=None,
                      csv_dest=None,
                      oids=(5537, 5547),
                      keep_in_between=False,
                      frames_before=None,
                      frames_after=None,
                      csv_cols=('condition', 'lock_object', 'framenumber', 't_sec', 't_nsec', 'exp_uuid')):
    """Combine-independent selection of object ids in csv and simple-flydra files."""
    # Most probably there is something like this in flydra
    # Here we do not even bother to use pytables or be nice in memory
    if keep_in_between or frames_before is not None or frames_after is not None:
        raise NotImplementedError('This would surely be convenient and it is should be easy to do...')
    import h5py
    with h5py.File(h5_file, 'r') as h5:
        # obj_id -> (first_timestamp_secs, first_timestamp_nsecs)
        starts = pd.DataFrame(h5['trajectory_start_times'][:])
        # obj_id -> (obj_id, framenumber, x, y, z)
        trajs = pd.DataFrame(h5['trajectories'][:])

        if h5_dest is None:
            h5_dest = op.join(op.dirname(h5_file), 'slim-' + op.basename(h5_file))

        with h5py.File(h5_dest, 'w') as h5_slim:
            h5_slim['experiment_info'] = h5['experiment_info'][:]  # uuid
            h5_slim['trajectories'] = trajs[trajs['obj_id'].isin(oids)].to_records()
            for k, v in h5['trajectories'].attrs.iteritems():
                h5_slim['trajectories'].attrs[k] = v
            h5_slim['trajectory_start_times'] = starts[starts['obj_id'].isin(oids)].to_records()
            for k, v in h5['trajectory_start_times'].attrs.iteritems():
                h5_slim['trajectory_start_times'].attrs[k] = v

    csv = pd.read_csv(csv_file)
    csv = csv[csv['lock_object'].isin(oids)]
    if csv_dest is None:
        csv_dest = op.join(op.dirname(csv_file), 'slim-' + op.basename(csv_file))
    if csv_cols is None:
        csv.to_csv(csv_dest, index=False)
    else:
        csv[list(csv_cols)].to_csv(csv_dest, index=False)
    # TODO: allow to add number of frames before and after


# --- tests for some combine functionality
#
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
        traj_datatypes, traj_start_datatypes = analysislib.combine.simple_flydra_datatypes()

        #we need to accumulate trajectory data from all flies (obj_ids)
        traj_data = {k[0]:[] for k in traj_datatypes}
        traj_starts_data = {k[0]:[] for k in traj_start_datatypes}

        #create a fake csv
        log_state = ("rotation_rate","ratio")
        log = nodelib.log.CsvLogger(fname=csv_fname,
                                    wait=False, debug=False, warn=False,
                                    state=log_state)

        cond_obj = Condition(value=1)
        cond_obj.name = 'test'

        log.condition = cond_obj
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
                        xfilt='none',
                        yfilt='none',
                        zfilt='none',
                        vfilt='none',
                        rfilt='none')

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


class TestCombineNonContiguous(unittest.TestCase):
    """
    Tests that combine separates correctly trajectories where  the same object_id
    contains trials with the same condition appearing several times.
    These happen when trials are long, for example within fish experiments.

    It also tests when the same condition interval contains different trials
    with the same oid, which happens everywhere.
    """

    def _combine_from_csv_h5(self, csv, h5):
        # combine the test csv/h5
        combine = get_combiner_for_csv(csv)
        combine.calc_turn_stats = False
        combine.calc_linear_stats = False
        combine.calc_angular_stats = False
        _, args = analysislib.args.get_default_args(xfilt='none',
                                                    yfilt='none',
                                                    zfilt='none',
                                                    vfilt='none',
                                                    rfilt='none',
                                                    lenfilt=0)

        combine.add_csv_and_h5_file(csv_fname=csv,
                                    h5_file=h5,
                                    args=args)
        _quiet(combine)
        return combine, csv, h5

    def _combine_inter_condition(self):
        """Returns a combine object useful for testing proper splitting of same-oid trajectories spanning
        more than one block with the same condition."""

        MAX_TEST_UUID = '6d7142fc643d11e4be3d60a44c2451e5'

        DATA_ROOT = op.abspath(op.join(op.dirname(__file__), 'data', 'contiguous', MAX_TEST_UUID))
        csv = op.join(DATA_ROOT, 'data.csv')
        h5 = op.join(DATA_ROOT, 'data.simple_flydra.h5')

        # generate small-test-data from real data
        if not op.isfile(csv) or not op.isfile(h5):
            # generate smaller test files we can fit in our repo
            combine = get_combiner_for_uuid(MAX_TEST_UUID)
            combine.calc_turn_stats = False
            combine.calc_linear_stats = False
            combine.calc_angular_stats = False
            combine.add_from_uuid(MAX_TEST_UUID,
                                  xfilt='none',
                                  yfilt='none',
                                  zfilt='none',
                                  vfilt='none',
                                  rfilt='none')
            analysislib.combine.combine2h5csv(combine,
                                              tempdir=DATA_ROOT,
                                              columns_for_csv=('stim_x',))

        return self._combine_from_csv_h5(csv, h5)

    def _combine_intra_condition(self, continuous_oids, with_markers):
        # 'non-contiguous-same-oid-consecutive-no-markers.csv'
        # 'non-contiguous-same-oid-consecutive-with-markers.csv'
        DATA_ROOT = op.abspath(op.join(op.dirname(__file__), 'data', 'contiguous'))
        base = 'non-contiguous-same-oid'
        if continuous_oids:
            base += '-consecutive'
        base += '-with-markers' if with_markers else '-no-markers'
        csv = op.join(DATA_ROOT, base + '.csv')
        h5 = op.join(DATA_ROOT, 'non-contiguous-same-oid.h5')
        return self._combine_from_csv_h5(csv, h5)

    def test_correct_intercondition_split(self):
        combine, csv, h5 = self._combine_inter_condition()
        results, dt = combine.get_results()
        self.assertAlmostEqual(dt, 0.01)
        self.assertEqual(len(results), 6)
        dfs = []
        for cond, condtrials in results.iteritems():
            for df in condtrials['df']:
                self.assertTrue(np.all(1 == df['framenumber'].diff().iloc[1:]))
                dfs.append(df)
        # there must be 20 trajectories
        self.assertEquals(20, len(dfs))
        # there must be 200 observations
        self.assertEquals(sum(map(len, dfs)), 200)

    def test_correct_intracondition_split(self):
        # Expectations...
        # There needs to be two trials for oid 5537: (595668, len=50 and 596002, len=180)
        # There needs to be one trial for oid 5547: (595784, len=213)
        def test_combine(combine, intermediate=True):
            valid_trials = (
                (5537, 595668, 50),
                (5537, 596002, 180),
            )
            if intermediate:
                valid_trials += ((5547, 595784, 213),)
            results, dt = combine.get_results()
            self.assertAlmostEqual(dt, 0.01)
            self.assertEqual(len(results), 1)  # 1 condition
            trials = results['gray.png/infinity07.svg/0.3/-5.0/0.1/0.18/0.2']
            for df, (x0, y0, obj_id, framenumber0, time0), uuid in zip(trials['df'],
                                                                       trials['start_obj_ids'],
                                                                       trials['uuids']):
                self.assertIn((obj_id, framenumber0, len(df)), valid_trials)
                self.assertEquals(uuid, 'e3ccf472a96711e48156bcee7bdac428')

        # With intermediate trial, with markers
        combine, _, _ = self._combine_intra_condition(continuous_oids=False, with_markers=True)
        test_combine(combine, intermediate=True)

        # With intermediate trial, without markers
        combine, _, _ = self._combine_intra_condition(continuous_oids=False, with_markers=False)
        test_combine(combine, intermediate=True)

        # Without intermediate trial, with markers
        combine, _, _ = self._combine_intra_condition(continuous_oids=True, with_markers=True)
        test_combine(combine, intermediate=False)

        # Without intermediate trial, without markers (using 10-frames heuristic)
        combine, _, _ = self._combine_intra_condition(continuous_oids=True, with_markers=False)
        test_combine(combine, intermediate=False)

if __name__ == '__main__':
    unittest.main()
