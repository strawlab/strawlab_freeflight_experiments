#!/usr/bin/env python
import numpy as np
import pandas as pd

import unittest

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')

import analysislib.combine as acomb
import analysislib.features as afeat

class TestFeaturesNumeric(unittest.TestCase):

    def test_velocity(self):
        EXPECTED_V = 100.0

        dt = 1/100.0
        x = np.array(range(100))
        y = np.array(range(100))
        z = np.array(range(100))
        df = pd.DataFrame({"x":x,"y":y,"z":z})

        m = afeat.MultiFeatureComputer('vx','vy','vz','velocity')
        m.process(df, dt)

        for col in ["vx","vy","vz","velocity"]:
            self.assertTrue(col in df)

        for i in ('vx','vy','vz'):
            #all velocity is the same
            self.assertTrue( np.allclose(df[i].values,EXPECTED_V) )

        #velocity in xy is 
        vxy = np.sqrt((EXPECTED_V**2) + (EXPECTED_V**2))
        self.assertTrue( np.allclose(df['velocity'].values,vxy) )

    def test_dtheta(self):
        # 2*pi in 10 sec --> 0.628319 rad/s
        phi = np.linspace(0,2*np.pi, 1000)
        t = np.linspace(0,10, 1000) 
        dt = t[1]-t[0] 

        x = np.cos(phi)
        y = np.sin(phi)
        z = y*0
        df = pd.DataFrame({'x': x, 'y': y, 'z':z})

        m = afeat.MultiFeatureComputer('dtheta')
        m.process(df, dt)

        dtheta = df['dtheta'].values

        self.assertTrue( np.allclose(dtheta[10:90],0.628319) )

class TestFeaturesAPI(unittest.TestCase):

    def test_set(self):
        m = afeat.MultiFeatureComputer('vx')
        self.assertEqual(repr(m.features), "[Measurement#col='x', VxFeature#col='vx']")
        self.assertEqual(len(m.features), 2)

        m.set_features()
        self.assertListEqual(m.features, [])

        m.set_features('vx')
        self.assertEqual(repr(m.features), "[Measurement#col='x', VxFeature#col='vx']")
        self.assertEqual(len(m.features), 2)

        #no-op
        m.add_feature('vx')
        self.assertEqual(repr(m.features), "[Measurement#col='x', VxFeature#col='vx']")
        self.assertEqual(len(m.features), 2)
        self.assertTupleEqual(m.get_columns_added(), ('vx',))

    def test_empty(self):
        m = afeat.MultiFeatureComputer()
        self.assertListEqual(m.features, [])

        m.add_feature('vx')
        self.assertEqual(repr(m.features), "[Measurement#col='x', VxFeature#col='vx']")
        self.assertEqual(len(m.features), 2)
        self.assertTupleEqual(m.get_columns_added(), ('vx',))

        #add same feature again
        m.add_feature('vx')
        self.assertEqual(repr(m.features), "[Measurement#col='x', VxFeature#col='vx']")
        self.assertEqual(len(m.features), 2)
        self.assertTupleEqual(m.get_columns_added(), ('vx',))

        #add an already added feature
        m.add_feature('x')
        self.assertEqual(repr(m.features), "[Measurement#col='x', VxFeature#col='vx']")
        self.assertEqual(len(m.features), 2)
        self.assertTupleEqual(m.get_columns_added(), ('vx',))

    def test_add(self):
        m = afeat.MultiFeatureComputer()
        self.assertListEqual(m.features, [])

        m.add_feature_by_column_added('vx')
        self.assertEqual(repr(m.features), "[Measurement#col='x', VxFeature#col='vx']")
        self.assertEqual(len(m.features), 2)
        self.assertTupleEqual(m.get_columns_added(), ('vx',))

        self.assertRaises(ValueError,m.add_feature_by_column_added,'NO_SUCH_COLUMN')
        self.assertRaises(ValueError,m.add_feature,'NO_SUCH_FEATURE')

    def test_more(self):
        m = afeat.MultiFeatureComputer('dtheta')
        self.assertEqual(len(m.features),6)
        self.assertTupleEqual(m.get_columns_added(), ('vx', 'vy', 'theta', 'dtheta'))

        #check the order of resolution is correct
        cls = afeat.get_feature_class('dtheta')
        self.assertTrue(isinstance(m.features[-1],cls))
        #after adding a new feature that is already computed
        m.add_feature('vx')
        self.assertEqual(len(m.features),6)
        self.assertTrue(isinstance(m.features[-1],cls))

        #after adding a new feature that is not yet computed
        m.add_feature('rcurve')
        self.assertEqual(len(m.features),7)


    def test_what(self):
        cls = afeat.get_feature_class('rcurve')
        self.assertEqual(cls().what().id(), "RCurveFeature#clip=(0, 1)#col='rcurve'#method='leastsq'#npts=10")

        m = afeat.MultiFeatureComputer('vx')
        self.assertEqual(m.what().id(),"MultiFeatureComputer#features=[VxFeature#col='vx']")


    def test_api(self):
        f = afeat.get_feature_class('rcurve')()
        self.assertTupleEqual(f.get_depends(),('x','y'))

        f = afeat.get_feature_class('vx')()
        self.assertTupleEqual(f.get_depends(),('x',))
        self.assertTupleEqual(f.get_adds(),('vx',))

    def test_all_features_have_correct_deps_and_adds(self):
        for cls in afeat.ALL_FEATURES:
            self.assertTrue(isinstance(cls.get_depends(),tuple))
            self.assertTrue(isinstance(cls.get_adds(),tuple))
            self.assertIsNotNone(cls.name)

    def test_lazy_compute(self):
        df = pd.DataFrame({'x':range(10),'y':range(10)})
        m = afeat.MultiFeatureComputer('vx','ratiouw')

        #we can compute vx, we can't compute ratiouw becuase ratio is missing
        computed,not_computed,missing = m.process(df,0.01)
        self.assertTupleEqual(computed,('vx',))
        self.assertTupleEqual(not_computed,('ratiouw',))
        self.assertTupleEqual(missing,('ratiouw','ratio'))

        #second computation computes nothing
        computed,not_computed,missing = m.process(df,0.01)
        self.assertTupleEqual(computed,())
        self.assertTupleEqual(not_computed,('vx','ratiouw'))
        self.assertTupleEqual(missing,('ratiouw','ratio'))

        #now we can add a ratio (required for ratiouw)
        df['ratio'] = range(10)
        computed,not_computed,missing = m.process(df,0.01)
        self.assertTupleEqual(computed,('ratiouw',))
        self.assertTupleEqual(not_computed,('vx',))
        self.assertTupleEqual(missing,())

class TestSomeFeatures(unittest.TestCase):

    def setUp(self):
        self._dt = 1/100.
        self._df = acomb._CombineFakeInfinity.get_fake_infinity(1,0,0,0,1,0,0,self._dt)

    def test_rcurve(self):
        f = afeat.get_feature('rcurve')
        rcurve_array = f.compute_from_df(self._df, self._dt, **f.DEFAULT_OPTS)

        df = self._df.copy()
        m = afeat.MultiFeatureComputer('rcurve')
        m.process(df, self._dt)

        self.assertTrue( np.allclose(df['rcurve'].values,rcurve_array) )

    def test_all_features_and_deps(self):
        for name in afeat.ALL_FEATURE_NAMES:
            if name in ('reprojection_error_smoothed', 'reprojection_error','inverse_dynamics'):
                #needs to load extra h5 data and uuid not known
                continue

            m = afeat.MultiFeatureComputer(name)
            df = self._df.copy()
            computed, not_computed, missing = m.process(df, self._dt)
            self.assertNotEqual(len(computed), 0)
            self.assertTupleEqual(missing,())

if __name__=='__main__':
    unittest.main()

