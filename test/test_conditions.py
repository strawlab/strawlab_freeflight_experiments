#!/usr/bin/env python
import os.path
from time import sleep
import unittest

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')

import strawlab_freeflight_experiments.conditions as sfe_conditions
import analysislib.fixes as afixes

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

    def test_duplicate_names(self):
        s = """
foo:
    bar: 1
foo2:
    bar: 1
"""
        self.assertRaises(ValueError, sfe_conditions.Conditions, s)

    def test_api(self):
        c = sfe_conditions.Conditions(open(self._yaml))

        c1 = c.first_condition()
        c2 = c.next_condition(c1)

        self.assertEqual(c1,c2)

    def test_normalise(self):
        orig = "checkerboard16.png/infinity.svg/+0.3/-10.0/0.1/0.20"
        fix = afixes.normalize_condition_string(orig)

        self.assertNotEqual(orig, fix)
        self.assertEqual(fix, "checkerboard16.png/infinity.svg/0.3/-10.0/0.1/0.2")

    def test_normalise_props(self):
        self.assertEqual(afixes.normalize_condition_string("+3"),"3")
        self.assertEqual(afixes.normalize_condition_string("-3"),"-3")
        self.assertEqual(afixes.normalize_condition_string("-3.0"),"-3.0")
        self.assertEqual(afixes.normalize_condition_string("+3.0"),"3.0")
        self.assertEqual(afixes.normalize_condition_string("+3.00"),"3.0")

    def test_randomisation(self):
        s = """
cond1:
    gain: 0.3
cond2:
    gain: 0.5
cond3:
    gain: 0.7
"""
        Conditions = sfe_conditions.Conditions

        def next_conditions(conds, n=9999):
            conditions = [conds.first_condition()]
            while len(conditions) < n:
                conditions.append(conds.next_condition(conditions[-1]))
            return [cond.name for cond in conditions]

        # sequential
        cond_names = next_conditions(Conditions(s, switch_order='seq'))
        self.assertEquals(['cond1', 'cond2', 'cond3'] * 3333, cond_names)

        # random is random, but controlled
        cond_names1 = next_conditions(Conditions(s, switch_order='fullrand', rng_seed=42))
        cond_names2 = next_conditions(Conditions(s, switch_order='fullrand', rng_seed=42))
        cond_names3 = next_conditions(Conditions(s, switch_order='fullrand', rng_seed=2147483647))
        self.assertEquals(cond_names1, cond_names2)
        self.assertNotEquals(cond_names1, cond_names3)

        # default is controlled (remove if we change the policy)
        cond_names1 = next_conditions(Conditions(s, switch_order='fullrand'))
        sleep(0.001)  # overkill
        cond_names2 = next_conditions(Conditions(s, switch_order='fullrand'))
        self.assertEquals(cond_names1, cond_names2)

        # proportions - serves as example in case we want to test non-uniform samplers
        self.assertAlmostEqual(cond_names1.count('cond1') / float(len(cond_names1)), 1/3., delta=0.01)
        self.assertAlmostEqual(cond_names1.count('cond2') / float(len(cond_names1)), 1/3., delta=0.01)
        self.assertAlmostEqual(cond_names1.count('cond3') / float(len(cond_names1)), 1/3., delta=0.01)

        # clock-based seed
        cond_names1 = next_conditions(Conditions(s, switch_order='fullrand', rng_seed=-1))
        sleep(0.001)  # overkill
        cond_names2 = next_conditions(Conditions(s, switch_order='fullrand', rng_seed=-1))
        self.assertNotEquals(cond_names1, cond_names2)

        # randstart
        cond_names1 = next_conditions(Conditions(s, switch_order='randstart', rng_seed=42))
        cond_names2 = next_conditions(Conditions(s, switch_order='randstart', rng_seed=42))
        self.assertNotEquals(['cond1', 'cond2', 'cond3'], cond_names1[:3])  # by chance, true, change seed if not
        self.assertEquals(cond_names1, cond_names2)
        self.assertEquals(cond_names1[0:3] * 3333, cond_names1)

    def test_compat(self):
        TO_TEST = {'rotation':['checkerboard16.png/infinity.svg/0.3/-10.0/0.1/0.2',
                               'gray.png/infinity07.svg/0.3/-5.0/0.1/0.18/0.2',
                               'checkerboard16.png/infinity07.svg/0.3/-5.0/0.1/0.18/0.2'],
                   'conflict':['checkerboard16.png/infinity07.svg/0.3/-5.0/0.1/0.18/0.2/justpost1.osg|-0.1|-0.1|0.0'],
                   'perturbation':['checkerboard16.png/infinity.svg/0.3/-10.0/0.1/0.2/multitone_rotation_rate|rudinshapiro2|1.8|3|1|5||0.4|0.46|0.56|0.96|1.0|0.0|0.06',
                                   'checkerboard16.png/infinity.svg/0.3/-10.0/0.1/0.2/step_rotation_rate|1.8|3|0.4|0.46|0.56|0.96|1.0|0.0|0.06']
        }

        for t,conds in TO_TEST.iteritems():
            for cond in conds:
                cc = sfe_conditions.ConditionCompat(cond)
                self.assertTrue(cc.is_type(t))
                self.assertEqual(cond, cc.to_slash_separated())




if __name__ == '__main__':
    unittest.main()
