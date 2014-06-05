#!/usr/bin/env python
import unittest

import numpy as np
import pandas as pd

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')
import analysislib.curvature as curve

import matplotlib.pyplot as plt

def _gen_known_correlation(n,ccef):
    #http://www.sitmo.com/article/generating-correlated-random-numbers/

    r = ccef

    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    y1 = r*x1 + (np.sqrt(1-(r**2))*x2)

    return x1,y1

class TestCorrelation(unittest.TestCase):

    def _get_test_data(self):
        n = 200000
        ccef = 0.66
        x,y = _gen_known_correlation(n,ccef)
        #fill with nans
        x[np.random.random(n) < 0.1] = np.nan
        y[np.random.random(n) < 0.1] = np.nan
        return x,y,ccef

    def test_shifted(self):
        x,y,ccef = self._get_test_data()

        #check that nans are removed
        cleanx,cleany,ccef2 = curve.calculate_correlation_and_remove_nans(x,y)
        self.assertAlmostEqual(ccef,ccef2,2)

        latencies = (5,10,20,80,200)
        for l in latencies:
            #shift
            x2 = x[l:]
            y2 = y[:-l]

            self.assertNotEqual(len(x),len(x2))

            res = curve.plot_correlation_latency_sweep(None,x2,y2,"x2","y2",1/100.0, latencies=latencies)
            self.assertAlmostEqual(ccef,res[l],2)

    def test_plot_shifted(self):
        x,y,ccef = self._get_test_data()
        #shift
        x2 = x[80:]
        y2 = y[:-80]

        fig = plt.figure()
        curve.plot_correlation_latency_sweep(fig,x2,y2,"x2","y2",1/100.0,hist2d=True)
        fig = plt.figure()
        curve.plot_correlation_latency_sweep(fig,x2,y2,"x2","y2",1/100.0,hist2d=False)
        print fig.get_children()

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

if __name__=='__main__':
    unittest.main()

