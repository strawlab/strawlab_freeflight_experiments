#!/usr/bin/env python
import unittest

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')
import analysislib.combine
import analysislib.args
import analysislib.util
import analysislib.fixes
import autodata.files

class TestCombineFake(unittest.TestCase):

    def setUp(self):
        self.uuid = "2a8386e0dd1911e3bd786c626d3a008a"
        fm = autodata.files.FileModel()
        fm.select_uuid(self.uuid)
        self.csv_file = fm.get_file_model("*.csv").fullpath
        self.h5_file = fm.get_file_model("simple_flydra.h5").fullpath

    def test_fix_condition(self):
        fu = analysislib.fixes.load_fixups(csv_file=self.csv_file,
                                           h5_file=self.h5_file)
        self.assertIsNotNone(fu)
        self.assertTrue(fu.active)
        self.assertTrue(fu.should_fix_condition)

        
        before = "checkerboard16.png/infinity.svg/+0.3/-10.0/0.1/0.20/chirp_linear|1.8|3|1.0|5.0|0.4|0.46|0.56|0.96|1.0|0.0|0.06"
        after = "checkerboard16.png/infinity.svg/+0.3/-10.0/0.1/0.20/chirp_rotation_rate|linear|1.8|3|1.0|5.0|0.4|0.46|0.56|0.96|1.0|0.0|0.06"

        a2 = fu.fix_condition(before)
        self.assertEqual(after, a2)

if __name__=='__main__':
    unittest.main()
