#!/usr/bin/env python
import unittest
import tempfile

import numpy as np
import pandas as pd

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')
import analysislib.combine
import analysislib.curvature as curve
import analysislib.args

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

    def _get_combine_data(self):
        tdir = tempfile.mkdtemp()
        combine = analysislib.combine._CombineFakeInfinity(nconditions=2,ntrials=20,ninfinity=7)
        parser,args = analysislib.args.get_default_args(
                outdir=tdir,
                lenfilt=3.5
        )
        combine.add_from_args(args)

        correlations = (('rotation_rate','dtheta'),)
        correlation_options = {"rotation_rate:dtheta":{"range":[[-1.45,1.45],[-10,10]]},
                               "latencies":set(range(0,40,2) + [40,80]),
                               "latencies_to_plot":(0,2,5,8,10,15,20,40,80),
        }

        return args, combine, correlations, correlation_options

    def test_correlation_plot(self):

        args, combine, correlations, correlation_options = self._get_combine_data()

        res = curve.plot_correlation_analysis(args, combine, correlations, correlation_options)

        for c in combine.get_conditions():
            ccef = res[c]["rotation_rate:dtheta"]
            self.assertAlmostEqual(ccef,1.0,1)

    def test_correlation_plot_with_impossible_correlations(self):

        # integration test: we get no result if all the time-series are constant
        # (it is hard to test this better given the current design)
        args, combine, correlations, correlation_options = self._get_combine_data()
        cond0, cond1 = combine.get_conditions()
        for df in combine.get_results()[0][cond0]['df']:
                df['rotation_rate'] = np.zeros(len(df))

        res = curve.plot_correlation_analysis(args, combine, correlations, correlation_options)

        self.assertEquals(len(res[cond0]), 0)
        self.assertEquals(len(res[cond1]), 1)
        self.assertAlmostEqual(res[cond1]['rotation_rate:dtheta'], 1.0, 1)

    def test_correlations(self):
        n = 100000
        ccef = 0.66

        x,y = _gen_known_correlation(n,ccef)
        df = pd.DataFrame({"x":x,"y":y})

        ccef1 = curve._correlate(df,"x","y")
        self.assertAlmostEqual(ccef,ccef1,2) #2 decimal places

        #fill with nans
        x2 = x.copy(); y2 = y.copy()
        x2[np.random.random(n) < 0.1] = np.nan
        y2[np.random.random(n) < 0.1] = np.nan
        df2 = pd.DataFrame({"x":x2,"y":y2})

        #check that nans are removed
        ccef2 = curve._correlate(df2,"x","y")
        self.assertAlmostEqual(ccef,ccef2,2)

        # this just tests that pandas does the right thing with constant arrays - a reminder
        df = pd.DataFrame({"x": np.ones(100), "y": np.linspace(0, 1, num=100)})
        self.assertTrue(np.isnan(curve._correlate(df, 'x', 'y')))
        self.assertTrue(np.isnan(curve._correlate(df, 'y', 'x')))



if __name__=='__main__':
    unittest.main()