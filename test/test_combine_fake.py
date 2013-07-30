#!/usr/bin/env python
import os.path
import sys
import numpy as np
import unittest
import collections

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')
import analysislib.combine
import analysislib.args

class TestCombineFake(unittest.TestCase):

    def setUp(self):
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

        self.assertEqual(self.combine.get_one_result(0), None)

        df,dt,(x0,y0,obj_id,framenumber0,time0) = self.combine.get_one_result(1)
        self._check_oid1(df,dt,x0,y0,obj_id,framenumber0,time0)

        self.assertEqual(time0, self.combine._t0 + self.combine._dt)

    def test_custom_filter(self):
        c1 = analysislib.combine._CombineFakeInfinity(nconditions=1,ntrials=1)
        c1.add_custom_filter("df[(df['ratio'] > 0.2) & (df['ratio'] < 0.8)]", 50)
        c1.add_from_args(self.args)

        #the filtered dataframe should be shorter than the original one
        df,dt,(x0,y0,obj_id,framenumber0,time0) = c1.get_one_result(1)
        self.assertLess(len(df), len(self.df0))

    def test_load_from_args(self):
        c1 = analysislib.combine._CombineFakeInfinity(nconditions=1,ntrials=1)
        parser,args = analysislib.args.get_default_args(
                outdir='/tmp/',
                show='--show' in sys.argv,
                customfilt="df[(df['ratio'] > 0.2) & (df['ratio'] < 0.8)]",
                customfilt_len=50,
                lenfilt=1
        )
        c1.add_from_args(args)

        #the filtered dataframe should be shorter than the original one
        df,dt,(x0,y0,obj_id,framenumber0,time0) = c1.get_one_result(1)
        self.assertLess(len(df), len(self.df0))

if __name__=='__main__':
    unittest.main()

