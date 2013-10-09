#!/usr/bin/env python
import os.path
import sys
import numpy as np
import unittest
import collections
import tempfile
import tarfile
import shutil

import roslib
import roslib.packages
roslib.load_manifest('strawlab_freeflight_experiments')
import analysislib.combine
import analysislib.util as autil

class TestDataStore(unittest.TestCase):

    def setUp(self):
        self._uuid = '0'*32
        self._tdir = tempfile.mkdtemp()
        self._ddir = os.path.join(self._tdir, self._uuid)
        os.makedirs(self._ddir)
        #make autodata look in the tempdir for file
        os.environ['FLYDRA_AUTODATA_BASEDIR'] = self._tdir

        #extract out trajectories there
        testdata = os.path.join(roslib.packages.get_pkg_dir('strawlab_freeflight_experiments'),
                           'test','20131004_161625.tar.bz2')

        t = tarfile.open(testdata,'r:bz2')
        t.extractall(self._ddir)

    def testLoad(self):
        combine = autil.get_combiner("rotation.csv")
        combine.add_from_uuid(self._uuid)

        n = combine.get_total_trials()
        self.assertEqual(n, 4)

        df,dt,(x0,y0,obj_id,framenumber0,time0) = combine.get_one_result(5)

    def tearDown(self):
        del os.environ['FLYDRA_AUTODATA_BASEDIR']
        shutil.rmtree(self._tdir)
        

if __name__=='__main__':
    unittest.main()

