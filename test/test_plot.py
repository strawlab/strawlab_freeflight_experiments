#!/usr/bin/env python
import os.path
import sys
import numpy as np
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__),'..','nodes'))
import conflict

import roslib
roslib.load_manifest('flycave')
import analysislib.combine
import analysislib.plots

class TestCombine(unittest.TestCase):
    def test_nloops(self):

        combine = analysislib.combine.CombineH5WithCSV(
                                conflict.Logger,
                                "ratio","rotation_rate",
                                debug=False,
        )
        combine.add_from_uuid("0aba1bb0ebc711e2a2706c626d3a008a", "conflict.csv", frames_before=0)

        df,dt,(x0,y0,obj_id,framenumber0,time0) = combine.get_one_result(422)

        self.assertEqual(analysislib.plots._calculate_nloops(df), 7)

if __name__=='__main__':
    unittest.main()

