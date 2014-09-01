#!/usr/bin/env python
import os.path
import sys
import numpy as np
import unittest
import tempfile

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')

import analysislib.combine
import analysislib.args
import analysislib.plots as aplt
import analysislib.util as autil

_me_exec = os.path.basename(sys.argv[0])

def _quiet(combine):
    combine.disable_warn()
    combine.disable_debug()

class TestPlot(unittest.TestCase):

    def _hplot(self, combine, args):
        ncond = combine.get_num_conditions()
        aplt.plot_histograms(combine, args,
                             figncols=ncond)

    def _tplot(self, combine, args):
        ncond = combine.get_num_conditions()
        aplt.plot_traces(combine, args,
                    figncols=ncond,
                    in3d=False,
                    show_starts=True,
                    show_ends=True)

    def _get_fake_tmpdir_combine(self):
        tdir = tempfile.mkdtemp()
        combine = analysislib.combine._CombineFakeInfinity(nconditions=1,ntrials=1,ninfinity=7)
        parser,args = analysislib.args.get_default_args(
                outdir=tdir,
                lenfilt=3.5
        )
        combine.add_from_args(args)
        return combine,args,tdir

    def test_plot_heuristics(self):
        combine,args,tdir = self._get_fake_tmpdir_combine()
        df,dt,(x0,y0,obj_id,framenumber0,time0) = combine.get_one_result(1)
        self.assertEqual(analysislib.plots._calculate_nloops(df), 6)

    def test_plots(self):
        combine,args,tdir = self._get_fake_tmpdir_combine()

        self.assertEqual(os.path.join(tdir,_me_exec,'test'), combine.fname)

        self._tplot(combine, args)
        self.assertTrue(os.path.isfile(combine.fname + ".traces.png"))

        self._hplot(combine, args)
        self.assertTrue(os.path.isfile(combine.fname + ".hist.png"))

    def test_misc_plots(self):
        combine,args,tdir = self._get_fake_tmpdir_combine()

        aplt.save_args(combine, args)
        self.assertTrue(os.path.isfile(os.path.join(combine.plotdir,"README")))

        aplt.save_results(combine, args)
        self.assertTrue(os.path.isfile(os.path.join(combine.plotdir,"data.json")))
        self.assertTrue(os.path.isfile(os.path.join(combine.plotdir,"data.pkl")))


if __name__=='__main__':
    unittest.main()

