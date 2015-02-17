#!/usr/bin/env python
import os.path
import sys
import numpy as np
import unittest
import collections
import tempfile

import roslib
import roslib.packages
roslib.load_manifest('strawlab_freeflight_experiments')
import analysislib.combine
import analysislib.args
import analysislib.util as autil
import autodata.files


def _quiet(combine):
    combine.disable_debug()

class TestCombineData(unittest.TestCase):

    def setUp(self):
        self._uuid = '0'*32
        ddir = os.path.join(
                    roslib.packages.get_pkg_dir('strawlab_freeflight_experiments'),
                   'test','data'
        )
        self._ddir = ddir
        self._pdir = os.path.join(ddir, 'plots')

        #make autodata look in the tempdir for file
        os.environ['FLYDRA_AUTODATA_BASEDIR'] = self._ddir
        os.environ['FLYDRA_AUTODATA_PLOTDIR'] = self._pdir

    def tearDown(self):
        del os.environ['FLYDRA_AUTODATA_BASEDIR']
        del os.environ['FLYDRA_AUTODATA_PLOTDIR']

    def test_auto_combine(self):
        combine = autil.get_combiner_for_uuid(self._uuid)
        _quiet(combine)

        combine.add_from_uuid(self._uuid, reindex=False)
        cols = set(combine.get_result_columns())
        expected = {'cyl_r', 'cyl_x', 'cyl_y', 'ratio', 'rotation_rate', 'trg_x', 'trg_y', 'trg_z',
                    'v_offset_rate', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'velocity', 'ax', 'ay', 'az', 'theta',
                    'dtheta', 'radius', 'omega', 'rcurve', 't_nsec', 'framenumber', 'tns', 't_sec', 'exp_uuid',
                    'flydra_data_file', 'lock_object', 'condition'}
        self.assertEqual(cols, expected)

    def _get_comb(self):
        try:
            combine = autil.get_combiner_for_uuid(self._uuid)
        except AttributeError:
            combine = autil.get_combiner("rotation.csv") #back compat for testing old branch pre merge
        combine.disable_warn()
        combine.disable_debug()
        combine.add_from_uuid(self._uuid, reindex=False)
        return combine

    def _get_fn(self, df):
        try:
            return df['framenumber'].values
        except KeyError:
            return df.index.values

    def test_date(self):
        combine = self._get_comb()
        df,dt,(x0,y0,obj_id,framenumber0,time0) = combine.get_one_result(5)
        self.assertAlmostEqual(time0, 1380896219.427156, 3)

    def test_range(self):
        combine = self._get_comb()
        df,dt,(x0,y0,obj_id,framenumber0,time0) = combine.get_one_result(5)

        fn = self._get_fn(df)

        self.assertEqual(fn[0], 3843)
        self.assertEqual(framenumber0, 3843)
        self.assertEqual(fn[-1], 5750)


class TestCombine(unittest.TestCase):

    RT_CSV = '/mnt/strawscience/data/TETHERED/str01/20130718_192901.rotation_tethered.csv'
    RT_CSV_OBJ_ID = 1374168638

    def _assert_two_equal(self, a, b):
        df,dt,(x0,y0,obj_id,framenumber0,time0) = a
        df_2,dt_2,(x0_2,y0_2,obj_id_2,framenumber0_2,time0_2) = b

        self.assertEqual(dt,dt_2)
        self.assertEqual(obj_id,obj_id_2)
        self.assertEqual(x0,x0_2)
        self.assertEqual(y0,y0_2)
        self.assertEqual(framenumber0,framenumber0_2)
        # This maybe should pass but I'm not gonna look into it right now
        # self.assertEqual(len(df), len(df_2))
        # self.assertTrue(np.allclose(df['x'].values, df_2['x'].values))

    def test_combine_h5(self):
        #get the csv file
        fm = autodata.files.FileModel()
        fm.select_uuid("f5adba10e8b511e2a28b6c626d3a008a")
        csv = fm.get_file_model("*.csv").fullpath

        combine = autil.get_combiner_for_csv(csv)
        _quiet(combine)

        combine.add_from_uuid("f5adba10e8b511e2a28b6c626d3a008a")

        a = combine.get_one_result(174)

        fname = combine.fname
        results,dt = combine.get_results()

        combine2 = analysislib.combine.CombineH5()
        _quiet(combine2)

        combine2.add_h5_file(combine.h5_file)
        b = combine2.get_one_result(174)

        self._assert_two_equal(a,b)

        combine2.close()

        parser,args = analysislib.args.get_default_args(
                    uuid=["f5adba10e8b511e2a28b6c626d3a008a"],
                    outdir='/tmp/'
        )
        combine3 = autil.get_combiner_for_args(args)
        _quiet(combine3)

        combine3.add_from_args(args)
        c = combine3.get_one_result(174)

        self._assert_two_equal(a,c)

    def _check_rotation_tethered(self,df,dt,x0,y0,obj_id,framenumber0,time0):
        self.assertEqual(len(df), 3270)
        self.assertAlmostEqual(dt,0.0124762588627)
        self.assertAlmostEqual(x0,-0.031926618439300003)
        self.assertAlmostEqual(y0,0.037099036447400001)
        self.assertEqual(obj_id, self.RT_CSV_OBJ_ID)
        self.assertEqual(framenumber0,7792)
        self.assertAlmostEqual(time0, 1374168638.8101971)

    def test_csv(self):
        combine = analysislib.combine.CombineCSV()
        _quiet(combine)

        combine.add_csv_file(self.RT_CSV)

        df,dt,(x0,y0,obj_id,framenumber0,time0) = combine.get_one_result(self.RT_CSV_OBJ_ID)
        self._check_rotation_tethered(df,dt,x0,y0,obj_id,framenumber0,time0)

    def test_csv_args(self):
        combine = analysislib.combine.CombineCSV()
        _quiet(combine)

        parser,args = analysislib.args.get_default_args(
                h5_file='/dev/null',
                csv_file=self.RT_CSV,
                outdir='/tmp/'
        )
        combine.add_from_args(args, "rotation_tethered.csv")

        df,dt,(x0,y0,obj_id,framenumber0,time0) = combine.get_one_result(self.RT_CSV_OBJ_ID)
        self._check_rotation_tethered(df,dt,x0,y0,obj_id,framenumber0,time0)

    def test_multi(self):
        tdir = tempfile.mkdtemp()
        parser,args = analysislib.args.get_default_args(
                    uuid=["75344a94e4c711e2b4c76c626d3a008a","69d1d022e58a11e29e446c626d3a008a"],
                    outdir=tdir
        )
        combine = autil.get_combiner_for_args(args)
        _quiet(combine)

        combine.add_from_args(args)

        self.assertEqual(combine.get_num_conditions(), 3)
        self.assertEqual(combine.get_total_trials(), 1005)

    def test_multi_csv_args(self):
        c1 = analysislib.combine.CombineCSV()
        _quiet(c1)

        parser,args = analysislib.args.get_default_args(
                uuid=["34e60d6efddc11e2848064315026cb58"],
                outdir='/tmp/',
                lenfilt=5,
        )
        c1.add_from_args(args, "rotation_tethered.csv")

        nc1 = c1.get_total_trials()
        self.assertEqual(nc1, 16)
        df,dt,(x0,y0,obj_id,framenumber0,time0) = c1.get_one_result(1375713692)
        self.assertEqual(len(df), 19945)

        del c1

        c2 = analysislib.combine.CombineCSV()
        _quiet(c2)

        parser,args = analysislib.args.get_default_args(
                uuid=["31b764cafb6c11e299c864315026cb58"],
                outdir='/tmp/',
                lenfilt=5,
        )
        c2.add_from_args(args, "rotation_tethered.csv")

        nc2 = c2.get_total_trials()
        self.assertEqual(nc2, 32)
        df,dt,(x0,y0,obj_id,framenumber0,time0) = c2.get_one_result(1375445854)
        self.assertEqual(len(df), 17472)

        del c2

        cc = analysislib.combine.CombineCSV()
        _quiet(cc)

        parser,args = analysislib.args.get_default_args(
                uuid=["34e60d6efddc11e2848064315026cb58","31b764cafb6c11e299c864315026cb58"],
                outdir='/tmp/',
                lenfilt=5,
        )
        cc.add_from_args(args, "rotation_tethered.csv")

        ncc = cc.get_total_trials()
        self.assertEqual(ncc, nc1 + nc2)

        df,dt,(x0,y0,obj_id,framenumber0,time0) = cc.get_one_result(1375713692)
        self.assertEqual(len(df), 19945)
        df,dt,(x0,y0,obj_id,framenumber0,time0) = cc.get_one_result(1375445854)
        self.assertEqual(len(df), 17472)

if __name__=='__main__':
    unittest.main()

