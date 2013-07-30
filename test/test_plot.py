#!/usr/bin/env python
import os.path
import sys
import numpy as np
import unittest
import tempfile

sys.path.append(os.path.join(os.path.dirname(__file__),'..','nodes'))
import conflict
import rotation

import roslib
roslib.load_manifest('flycave')
import analysislib.combine
import analysislib.args
import analysislib.plots as aplt

class TestCombine(unittest.TestCase):

    def _hplot(self, combine, args):
        ncond = combine.get_num_conditions()
        aplt.plot_histograms(combine, args,
                    figsize=(5*ncond,5),
                    fignrows=1, figncols=ncond,
                    radius=[0.5],
                    name='%s.hist' % combine.fname)

    def _tplot(self, combine, args):
        ncond = combine.get_num_conditions()
        aplt.plot_traces(combine, args,
                    figsize=(5*ncond,5),
                    fignrows=1, figncols=ncond,
                    in3d=False,
                    radius=[0.5],
                    name='%s.traces' % combine.fname,
                    show_starts=True,
                    show_ends=True)

    def test_conflict(self):

        combine = analysislib.combine.CombineH5WithCSV(
                                conflict.Logger,
                                "ratio","rotation_rate",
                                debug=False,
        )
        combine.add_from_uuid("0aba1bb0ebc711e2a2706c626d3a008a", "conflict.csv", frames_before=0)

        df,dt,(x0,y0,obj_id,framenumber0,time0) = combine.get_one_result(422)

        self.assertEqual(analysislib.plots._calculate_nloops(df), 7)

    def test_rotation(self):
        tdir = tempfile.mkdtemp()

        combine = analysislib.combine.CombineH5WithCSV(
                                conflict.Logger,
                                "ratio","rotation_rate",
                                debug=False,
        )
        parser,args = analysislib.args.get_default_args(
                    uuid=["75344a94e4c711e2b4c76c626d3a008a"],
                    outdir=tdir
        )
        combine.add_from_args(args, "rotation.csv")

        self.assertEqual(os.path.join(tdir,'20130704_182833'), combine.fname)

        self.assertEqual(combine.get_num_conditions(), 3)
        self.assertEqual(combine.get_total_trials(), 481)

        self._tplot(combine, args)
        self.assertTrue(os.path.isfile(combine.fname + ".traces.png"))

        self._hplot(combine, args)
        self.assertTrue(os.path.isfile(combine.fname + ".hist.png"))

    def test_combine_rotation(self):
        tdir = tempfile.mkdtemp()

        combine = analysislib.combine.CombineH5WithCSV(
                                conflict.Logger,
                                "ratio","rotation_rate",
                                debug=False,
        )
        parser,args = analysislib.args.get_default_args(
                    uuid=["75344a94e4c711e2b4c76c626d3a008a","69d1d022e58a11e29e446c626d3a008a"],
                    outdir=tdir
        )
        combine.add_from_args(args, "rotation.csv")

        self.assertEqual(os.path.basename(combine.fname), "20130705_174835")

        self.assertEqual(combine.get_num_conditions(), 3)
        self.assertEqual(combine.get_total_trials(), 998)

        self._tplot(combine, args)
        self.assertTrue(os.path.isfile(combine.fname + ".traces.png"))

        self._hplot(combine, args)
        self.assertTrue(os.path.isfile(combine.fname + ".hist.png"))

    def test_combine_requires_outdir(self):
        parser,args = analysislib.args.get_default_args(
                    uuid=["75344a94e4c711e2b4c76c626d3a008a","69d1d022e58a11e29e446c626d3a008a"],
        )

        self.assertRaises(SystemExit, analysislib.args.check_args, parser, args)

if __name__=='__main__':
    unittest.main()

