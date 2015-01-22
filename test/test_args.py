#!/usr/bin/env python
import os.path
import sys
import numpy as np
import unittest
import tempfile

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')
import analysislib.args as aargs

class TestArgs(unittest.TestCase):

    def test_combine_requires_outdir(self):
        parser,args = aargs.get_default_args(
                    uuid=["75344a94e4c711e2b4c76c626d3a008a","69d1d022e58a11e29e446c626d3a008a"],
        )

        self.assertRaises(SystemExit, aargs.check_args, parser, args)

    def test_get_parser(self):
        p1 = aargs.get_parser("show")
        self.assertEqual(p1.get_default("show"), False)
        p2 = aargs.get_parser("zfilt",zfilt="none")
        self.assertEqual(p2.get_default("zfilt"), "none")

if __name__=='__main__':
    unittest.main()

