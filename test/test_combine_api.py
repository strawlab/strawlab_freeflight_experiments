#!/usr/bin/env python
import os.path
import sys
import numpy as np
import unittest
import collections

import roslib
import roslib.packages
roslib.load_manifest('strawlab_freeflight_experiments')
import analysislib.combine
import analysislib.args
import analysislib.util as autil
import autodata.files

def _quiet(combine):
    combine.disable_debug()


if not analysislib.combine.is_testing():
    raise Exception('Combine tests must run under nose or with the environment var NOSETEST_FLAG=1')


class TestCombineAPI(unittest.TestCase):

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

    def _get_combine(self):
        combine = autil.get_combiner_for_uuid(self._uuid)
        _quiet(combine)
        combine.add_from_uuid(self._uuid, reindex=False)
        return combine

    def test_auto_combine(self):
        combine = self._get_combine()

        self.assertEqual(combine.get_num_conditions(), 1)
        self.assertEqual(combine.get_total_trials(), 5)
   

if __name__=='__main__':
    unittest.main()

