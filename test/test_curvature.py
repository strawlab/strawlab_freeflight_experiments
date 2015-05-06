#!/usr/bin/env python
import unittest

import numpy as np
import pandas as pd

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')
import analysislib.curvature as curve

import matplotlib.pyplot as plt

class TestCurvature(unittest.TestCase):

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

