#!/usr/bin/env python
import unittest

import numpy as np
import pandas as pd

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')
import analysislib.curvature as curve

import matplotlib.pyplot as plt

class TestCurvature(unittest.TestCase):

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

    def test_dtheta(self):
        # 2*pi in 10 sec --> 0.628319 rad/s
        phi = np.linspace(0,2*np.pi, 1000)
        t = np.linspace(0,10, 1000) 
        dt = t[1]-t[0] 

        x = np.cos(phi)
        y = np.sin(phi)
        z = y*0
        df = pd.DataFrame({'x': x, 'y': y, 'z':z})

        curve.calc_velocities(df, dt)
        curve.calc_angular_velocities(df, dt)

        dtheta = df['dtheta'].values

        self.assertTrue( np.allclose(dtheta[10:90],0.628319) )

if __name__=='__main__':
    unittest.main()

