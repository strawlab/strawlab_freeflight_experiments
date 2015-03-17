#!/usr/bin/env python
import os.path
from time import sleep
import unittest

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')

import strawlab_freeflight_experiments.conditions as sfe_conditions
import strawlab_freeflight_experiments.perturb as sfe_perturb
import analysislib.fixes as afixes

class TestConditions(unittest.TestCase):

    def setUp(self):
        ddir = os.path.join(
                    roslib.packages.get_pkg_dir('strawlab_freeflight_experiments'),
                   'test','data'
        )
        self._yaml = os.path.join(ddir,'conditions.yaml')

    def test_load_time_perturb(self):
        s = """
perturbation_mtrs2_18_3s_5hz:
    cylinder_image: checkerboard16.png
    svg_path: infinity.svg
    gain: 0.3
    radius_when_locked: -10.0
    advance_threshold: 0.1
    z_gain: 0.20
    perturb_desc: multitone_rotation_rate|rudinshapiro2|1.8|3|1|5|
    perturb_criteria: t5
"""

        c = sfe_conditions.Conditions(s).first_condition()
        self.assertTrue(c.is_type('perturbation'))

        obj = sfe_perturb.get_perturb_object_from_condition(c)
        self.assertTrue(isinstance(obj,sfe_perturb.PerturberMultiTone))
        self.assertEqual(obj.criteria_type, sfe_perturb.Perturber.CRITERIA_TYPE_TIME)

    def test_load_ratio_perturb_old(self):
        s = """
perturbation_mtrs2_18_3s_5hz:
    cylinder_image: checkerboard16.png
    svg_path: infinity.svg
    gain: 0.3
    radius_when_locked: -10.0
    advance_threshold: 0.1
    z_gain: 0.20
    perturb_desc: multitone_rotation_rate|rudinshapiro2|1.8|3|1|5||0.4|0.46|0.56|0.96|1.0|0.0|0.06
"""

        c = sfe_conditions.Conditions(s).first_condition()
        self.assertTrue(c.is_type('perturbation'))

        obj = sfe_perturb.get_perturb_object_from_condition(c)
        self.assertTrue(isinstance(obj,sfe_perturb.PerturberMultiTone))
        self.assertEqual(obj.criteria_type, sfe_perturb.Perturber.CRITERIA_TYPE_RATIO)

        #old slash codepath
        cond = c.to_slash_separated()
        desc = cond.split('/')[-1]

        obj = sfe_perturb.get_perturb_object(desc)

        self.assertTrue(isinstance(obj,sfe_perturb.PerturberMultiTone))
        self.assertEqual(obj.criteria_type, sfe_perturb.Perturber.CRITERIA_TYPE_RATIO)

    def test_load_ratio_perturb_new(self):
        s = """
perturbation_mtrs2_18_3s_5hz:
    cylinder_image: checkerboard16.png
    svg_path: infinity.svg
    gain: 0.3
    radius_when_locked: -10.0
    advance_threshold: 0.1
    z_gain: 0.20
    perturb_desc: multitone_rotation_rate|rudinshapiro2|1.8|3|1|5|
    perturb_criteria: 0.4|0.46|0.56|0.96|1.0|0.0|0.06
"""

        c = sfe_conditions.Conditions(s).first_condition()
        self.assertTrue(c.is_type('perturbation'))

        obj = sfe_perturb.get_perturb_object_from_condition(c)
        self.assertTrue(isinstance(obj,sfe_perturb.PerturberMultiTone))
        self.assertEqual(obj.criteria_type, sfe_perturb.Perturber.CRITERIA_TYPE_RATIO)

if __name__ == '__main__':
    unittest.main()
