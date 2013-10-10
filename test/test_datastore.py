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
        self.uuid = '0'*32
        self.tdir = tempfile.mkdtemp()
        self.ddir = os.path.join(self.tdir, self.uuid)
        self.pdir = os.path.join(self.tdir, 'plots')
        os.makedirs(self.ddir); os.makedirs(self.pdir)
        #make autodata look in the tempdir for file
        os.environ['FLYDRA_AUTODATA_BASEDIR'] = self.tdir
        os.environ['FLYDRA_AUTODATA_PLOTDIR'] = self.pdir

        #extract out trajectories there
        testdata = os.path.join(roslib.packages.get_pkg_dir('strawlab_freeflight_experiments'),
                           'test','20131004_161625.tar.bz2')

        t = tarfile.open(testdata,'r:bz2')
        t.extractall(self.ddir)

        self.combine = autil.get_combiner("rotation.csv")
        self.combine.add_from_uuid(self.uuid)

    def testLoad(self):
        n = self.combine.get_total_trials()
        self.assertEqual(n, 4)
        df,dt,(x0,y0,obj_id,framenumber0,time0) = self.combine.get_one_result(5)

    def testFilenames(self):
        readme = self.combine.get_plot_filename("README")
        self.assertEqual(readme, os.path.join(self.pdir,self.uuid,'README'))

        fname = self.combine.fname
        self.assertEqual(fname, os.path.join(self.pdir,self.uuid,"20131004_161631"))

        plotdir = self.combine.plotdir
        self.assertEqual(plotdir, os.path.join(self.pdir,self.uuid)+"/")

    def tearDown(self):
        del os.environ['FLYDRA_AUTODATA_BASEDIR']
        del os.environ['FLYDRA_AUTODATA_PLOTDIR']
        shutil.rmtree(self.tdir)

if __name__=='__main__':
    unittest.main()

