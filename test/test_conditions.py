#!/usr/bin/env python
import os.path
import unittest

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')

import strawlab_freeflight_experiments.conditions as sfe_conditions

class TestConditions(unittest.TestCase):

    def setUp(self):
        ddir = os.path.join(
                    roslib.packages.get_pkg_dir('strawlab_freeflight_experiments'),
                   'test','data'
        )
        self._yaml = os.path.join(ddir,'conditions.yaml')

    def test_load(self):
        s = """
rotation_infinity:
    cylinder_image: checkerboard16.png
    svg_path: infinity.svg
    gain: 0.3
    radius_when_locked: -10.0
    advance_threshold: 0.1
    z_gain: 0.20
"""

        c1 = sfe_conditions.Conditions(s)
        self.assertEqual(len(c1), 1)

        c2 = sfe_conditions.Conditions(open(self._yaml))
        self.assertEqual(c1, c2)

    def test_api(self):
        c = sfe_conditions.Conditions(open(self._yaml))

        c1 = c.first_condition()
        c2 = c.next_condition(c1)

        self.assertEqual(c1,c2)

if __name__ == '__main__':
    unittest.main()
