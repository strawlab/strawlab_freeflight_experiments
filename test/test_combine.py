#!/usr/bin/env python
import os.path
import sys
import numpy as np
import unittest
import collections

sys.path.append(os.path.join(os.path.dirname(__file__),'..','nodes'))
import conflict

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')
import analysislib.combine
import analysislib.args

class TestCombine(unittest.TestCase):

    RT_CSV = '/mnt/strawarchive/data/TETHERED/str01/20130718_192901.rotation_tethered.csv'
    RT_CSV_OBJ_ID = 1374168638

    def _assert_two_equal(self, a, b):
        df,dt,(x0,y0,obj_id,framenumber0,time0) = a
        df_2,dt_2,(x0_2,y0_2,obj_id_2,framenumber0_2,time0_2) = b

        self.assertEqual(dt,dt_2)
        self.assertEqual(obj_id,obj_id_2)
        self.assertEqual(x0,x0_2)
        self.assertEqual(y0,y0_2)
        self.assertEqual(framenumber0,framenumber0_2)

    def test_combine_h5(self):
        combine = analysislib.combine.CombineH5WithCSV(
                                conflict.Logger,
                                "ratio","rotation_rate",
                                debug=False,
        )
        combine.add_from_uuid("f5adba10e8b511e2a28b6c626d3a008a", "conflict.csv", frames_before=0)
        a = combine.get_one_result(174)

        fname = combine.fname
        results,dt = combine.get_results()

        combine2 = analysislib.combine.CombineH5()
        combine2.add_h5_file(combine.h5_file)
        b = combine2.get_one_result(174)

        self._assert_two_equal(a,b)

        combine3 = analysislib.combine.CombineH5WithCSV(
                                conflict.Logger,
                                "ratio","rotation_rate",
                                debug=False,
        )
        parser,args = analysislib.args.get_default_args(
                    uuid=["f5adba10e8b511e2a28b6c626d3a008a"],
                    outdir='/tmp/'
        )
        combine3.add_from_args(args, "conflict.csv")
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
        combine.add_csv_file(self.RT_CSV)

        df,dt,(x0,y0,obj_id,framenumber0,time0) = combine.get_one_result(self.RT_CSV_OBJ_ID)
        self._check_rotation_tethered(df,dt,x0,y0,obj_id,framenumber0,time0)

    def test_csv_args(self):
        combine = analysislib.combine.CombineCSV()
        parser,args = analysislib.args.get_default_args(
                csv_file=self.RT_CSV,
                outdir='/tmp/'
        )
        combine.add_from_args(args, "rotation_tethered.csv")

        df,dt,(x0,y0,obj_id,framenumber0,time0) = combine.get_one_result(self.RT_CSV_OBJ_ID)
        self._check_rotation_tethered(df,dt,x0,y0,obj_id,framenumber0,time0)

    def test_multi_csv_args(self):
        c1 = analysislib.combine.CombineCSV()
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

