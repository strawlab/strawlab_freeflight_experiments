#!/usr/bin/env python
import unittest

import numpy as np
import pandas as pd

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')
import analysislib.curvature as curve

class TestCurvature(unittest.TestCase):

    def test_correlations(self):
        a = np.linspace(1,100)
        b = np.linspace(1,100) * 2
        cleana,cleanb,ccef = curve.calculate_correlation_and_remove_nans(a,b)

        self.assertEqual(ccef,1.0)
        self.assertTrue(np.allclose(a,cleana))
        self.assertTrue(np.allclose(b,cleanb))

        #fill with nans
        a2 = a.copy(); b2 = b.copy()
        a2[np.random.random(50) < 0.05] = np.nan
        b2[np.random.random(50) < 0.05] = np.nan
        self.assertTrue(np.isnan(a2).sum() > 0)

        #check that nans are removed
        cleana,cleanb,ccef = curve.calculate_correlation_and_remove_nans(a2,b2)
        self.assertEqual(ccef,1.0)
        self.assertTrue(len(cleana) < len(a2))
        self.assertTrue(len(cleanb) < len(b2))

    def test_velocity(self):
        EXPECTED_V = 100.0

        dt = 1/100.0
        x = np.array(range(100))
        y = np.array(range(100))
        z = np.array(range(100))
        df = pd.DataFrame({"x":x,"y":y,"z":z})

        cols = curve.calc_velocities(df, dt)
        self.assertEqual(cols, ["vx","vy","vz","velocity"])

        for i in ('vx','vy','vz'):
            #all velocity is the same
            self.assertTrue( np.allclose(df[i].values,EXPECTED_V) )

        #velocity in xy is 
        vxy = np.sqrt((EXPECTED_V**2) + (EXPECTED_V**2))
        self.assertTrue( np.allclose(df['velocity'].values,vxy) )

if __name__=='__main__':
    unittest.main()

