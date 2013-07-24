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
        args = analysislib.args.get_default_args(
                outdir='/tmp/'
        )
        self.combine.add_from_args(args)

    def _check_oid1(self, df,dt,x0,y0,obj_id,framenumber0,time0):
        self.assertEqual(dt, self.combine._dt)
        self.assertEqual(x0, 0.0)
        self.assertEqual(y0, 0.0)
        self.assertEqual(obj_id, 1)
        self.assertEqual(framenumber0, 1)

    def test_load(self):
        self.assertEqual(self.combine.get_one_result(0), None)

        df,dt,(x0,y0,obj_id,framenumber0,time0) = self.combine.get_one_result(1)
        self._check_oid1(df,dt,x0,y0,obj_id,framenumber0,time0)

        self.assertEqual(time0, self.combine._t0 + self.combine._dt)

    def test_load_from_args(self):
        combine = analysislib.combine._CombineFakeInfinity(nconditions=1)
        args = analysislib.args.get_default_args(
                outdir='/tmp/',
                show='--show' in sys.argv
        )
        combine.add_from_args(args)

        df,dt,(x0,y0,obj_id,framenumber0,time0) = combine.get_one_result(1)
        self._check_oid1(df,dt,x0,y0,obj_id,framenumber0,time0)

if __name__=='__main__':
    unittest.main()

