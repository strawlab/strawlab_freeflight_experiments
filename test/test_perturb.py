#!/usr/bin/env python
import os.path
from time import sleep
import unittest

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')

import strawlab_freeflight_experiments.conditions as sfe_conditions
import strawlab_freeflight_experiments.perturb as sfe_perturb
import analysislib.fixes as afixes
import analysislib.util as autil
import analysislib.perturb as aperturb

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

class TestExtractPerturbations(unittest.TestCase):

    def setUp(self):
        self._cond = 'checkerboard16.png/infinity.svg/0.3/3/-10.0/0.1/0.2/idinput_rotation_rate|sine|3|0|5|1.8||||1|0.4|0.46|0.56|0.96|1.0|0.0|0.06'
        self._uuid = 'b4208cdabc4411e49c956c626d3a008a'

    def _get_combine(self):
        combine = autil.get_combiner_for_uuid(self._uuid)
        combine.disable_debug()
        combine.add_from_uuid(self._uuid, reindex=False)
        return combine

    def test_collect_perturbation_traces(self):
        c = self._get_combine()

        perturbations, perturbation_objects = aperturb.collect_perturbation_traces(c, completion_threshold=0.5)
        pos = perturbation_objects[self._cond]
        phs = perturbations[self._cond]

        self.assertIsInstance(pos,sfe_perturb.PerturberIDINPUT)

        self.assertEqual(len(phs), 10)
        self.assertEqual(sum(len(ph.df) for ph in phs), 7419)
        self.assertEqual(sum(ph.completed for ph in phs), 7)

        #differet completion_thresh
        perturbations, perturbation_objects = aperturb.collect_perturbation_traces(c, completion_threshold=0.98)
        pos = perturbation_objects[self._cond]
        phs = perturbations[self._cond]

        self.assertEqual(len(phs), 10)
        self.assertEqual(sum(len(ph.df) for ph in phs), 7419)   #same as before
        self.assertEqual(sum(ph.completed for ph in phs), 4)    #less should complete


if __name__ == '__main__':
    unittest.main()
