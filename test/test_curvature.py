#!/usr/bin/env python
import unittest

import numpy as np
import pandas as pd

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')
import analysislib.curvature as curve

def _gen_known_correlation(n,ccef):
    #http://www.sitmo.com/article/generating-correlated-random-numbers/

    r = ccef

    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    y1 = r*x1 + (np.sqrt(1-(r**2))*x2)

    return x1,y1

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
        a2[np.random.random(50) < 0.1] = np.nan
        b2[np.random.random(50) < 0.1] = np.nan
        self.assertTrue(np.isnan(a2).sum() > 0)

        #check that nans are removed
        cleana,cleanb,ccef = curve.calculate_correlation_and_remove_nans(a2,b2)
        self.assertEqual(ccef,1.0)
        self.assertTrue(len(cleana) < len(a2))
        self.assertTrue(len(cleanb) < len(b2))

    def test_correlations_two(self):
        n = 100000
        ccef = 0.66

        x,y = _gen_known_correlation(n,ccef)

        _,_,ccef1 = curve.calculate_correlation_and_remove_nans(x,y)
        self.assertAlmostEqual(ccef,ccef1,2) #2 decimal places

        #fill with nans
        x2 = x.copy(); y2 = y.copy()
        x2[np.random.random(n) < 0.1] = np.nan
        y2[np.random.random(n) < 0.1] = np.nan

        #check that nans are removed
        _,_,ccef2 = curve.calculate_correlation_and_remove_nans(x2,y2)
        self.assertAlmostEqual(ccef,ccef2,2)

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

    def test_rcurve(self):
        dt = 1/100.0
        rad = 0.5
        theta = np.linspace(0, 2*np.pi, 100)
        x = rad*np.cos(theta)
        y = rad*np.sin(theta)

        #get the right radius using all the data
        self.assertEqual(curve.calc_circle_algebraic(x,y), rad)
        self.assertAlmostEqual(curve.calc_circle_leastsq(x,y), rad)

        df = pd.DataFrame({"x":x,"y":y,"z":np.zeros_like(theta)})
        for meth in ("algebraic","leastsq"):
            curve.calc_curvature(df, dt, NPTS=3, method=meth, clip=None, colname=meth)
            v = df[meth].values
            #no nans
            self.assertTrue( np.alltrue(~np.isnan(v)) )
            #estimate correct radius
            self.assertTrue( np.allclose(v,rad) )

        #two rad curve
        rad2 = 0.25
        theta = np.linspace(0, 2*np.pi, 100)
        x2 = rad2*np.cos(theta)
        y2 = rad2*np.sin(theta)
        half = len(x)//2
        x2[half:] = x[half:]
        y2[half:] = y[half:]

        df = pd.DataFrame({"x":x2,"y":y2,"z":np.zeros_like(theta)})
        for meth in ("algebraic","leastsq"):
            curve.calc_curvature(df, dt, NPTS=3, method=meth, clip=None, colname=meth)
            v = df[meth].values
            #no nans
            self.assertTrue( np.alltrue(~np.isnan(v)) )

            #use quarter not half for test because discontinuous where the 
            #two arrays have been joined
            #first quarter has small radius
            self.assertTrue( np.allclose(v[0:half//2],rad2) )
            #second quarter has beg radius
            self.assertTrue( np.allclose(v[-half//2:],rad) )

if __name__=='__main__':
    unittest.main()

