#!/usr/bin/env python
import os.path
import unittest

import roslib
import roslib.packages
roslib.load_manifest('strawlab_freeflight_experiments')
import analysislib.combine
import analysislib.util as autil
import strawlab.constants

class TestDataStore(unittest.TestCase):

    def setUp(self):
        self.uuid = '0'*32
        ddir = os.path.join(
                    roslib.packages.get_pkg_dir('strawlab_freeflight_experiments'),
                   'test','data'
        )
        self.ddir = os.path.join(ddir, self.uuid)
        self.pdir = os.path.join(ddir, 'plots')
        #make autodata look in the tempdir for file
        os.environ['FLYDRA_AUTODATA_BASEDIR'] = ddir
        os.environ['FLYDRA_AUTODATA_PLOTDIR'] = self.pdir

        self.combine = autil.get_combiner("rotation.csv")
        self.combine.add_from_uuid(self.uuid)

    def testLoad(self):
        n = self.combine.get_total_trials()
        self.assertEqual(n, 4)
        df,dt,(x0,y0,obj_id,framenumber0,time0) = self.combine.get_one_result(5)
        self.assertEqual(len(df), 1908)

    def testSingleLoad(self):
        df,dt = autil.get_one_trajectory(self.uuid, 5, "rotation.csv")
        self.assertEqual(len(df), 1908)

    def testFilenames(self):
        readme = self.combine.get_plot_filename("README")
        self.assertEqual(readme, os.path.join(self.pdir,self.uuid,'README'))

        fname = self.combine.fname
        self.assertEqual(fname, os.path.join(self.pdir,self.uuid,"20131004_161631"))

        plotdir = self.combine.plotdir
        self.assertEqual(plotdir, os.path.join(self.pdir,self.uuid))

    def tearDown(self):
        del os.environ['FLYDRA_AUTODATA_BASEDIR']
        del os.environ['FLYDRA_AUTODATA_PLOTDIR']

if __name__=='__main__':
    unittest.main()

