#!/usr/bin/env python
import matplotlib
matplotlib.use('agg')

import os.path
import unittest

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')

import numpy as np
import shapely.geometry as sg
from shapely.geometry.polygon import LinearRing, LineString

from flyflypath.euclid import Point2, LineSegment2, Vector2
from flyflypath.polyline import PolyLine2, ZeroLineSegment2
from flyflypath.transform import SVGTransform
from flyflypath.model import MovingPointSvgPath, HitManager, SvgError

import matplotlib.pyplot as plt
import flyflypath.mplview as pltpath

class TestFlyFlyPath(unittest.TestCase):

    def test_polyline(self):

        p00 = Point2(0,0)

        p0 = Point2(1,1)
        p1 = Point2(1,2)
        p1b = Point2(1,2)
        p2 = Point2(2,2)
        p4 = Point2(3,3)
        p4b = Point2(3,3)

        self.assertEqual(p1,p1b)

        polyl = PolyLine2(p0,p1,p1b,p2)

        self.assertEqual(polyl.p1, p0)

        self.assertEqual(polyl.length,2)

        self.assertEqual(polyl.along(0.5),p1)
        self.assertEqual(polyl.along(0.75),Point2(1.5,2))

        polyreverse = PolyLine2(p2,p1b,p1,p0)
        self.assertEqual(polyl.length,polyreverse.length)
        self.assertEqual(polyreverse.along(0.75),Point2(1,1.5))
        
        polya = PolyLine2(p0,p1,p1b,p2,p4,p4b)
        polyb = PolyLine2(p0,p1,p2,p4)
        
        self.assertEqual(polya.length,polyb.length)
        self.assertEqual(polya.num_segments,polyb.num_segments)
        
        self.assertEqual((p0-p1).magnitude(),1)

        self.assertEqual(ZeroLineSegment2().length,0)
        self.assertEqual(ZeroLineSegment2().p1,p00)
        self.assertEqual(ZeroLineSegment2().v.x,0)

        self.assertEqual(ZeroLineSegment2(p1).p1,p1)

        l0 = LineSegment2(p0,p1)
        self.assertEqual(l0.p1,p0)
        self.assertEqual(l0.p2,p1)

        l1 = LineSegment2(p1,p2)

        self.assertEqual(l0.length,1)
        self.assertEqual(l1.length,1)
        self.assertEqual(LineSegment2(p0,p2).length,np.sqrt(2))
        self.assertEqual((p2-p0).magnitude_squared(),2)
        
        tp0 = Point2(0.3,0.4)
        tp1 = Point2(1.2,2.2)
        tp2 = Point2(2,2)
        tp3 = Point2(2.8,2.2)
        tp4 = Point2(30,40)

        p = PolyLine2(p0,p1,p2,p4)

        pt,r =  p.connect(tp0)
        self.assertEqual(pt,Point2(1.00, 1.00))
        self.assertAlmostEqual(r,0)
        pt,r = p.connect(tp1)
        self.assertEqual(pt,Point2(1.20, 2.00))
        self.assertAlmostEqual(r,0.351471862576143)
        pt,r = p.connect(tp2)
        self.assertEqual(pt,Point2(2.00, 2.00))
        self.assertAlmostEqual(r,0.585786437626905)
        pt,r = p.connect(tp3)
        self.assertEqual(pt,Point2(2.50, 2.50))
        self.assertAlmostEqual(r,0.7928932188134525)
        pt,r = p.connect(tp4)
        self.assertEqual(pt,Point2(3.00, 3.00))
        self.assertAlmostEqual(r,1.0)

    def test_xform(self):
        XFORM = SVGTransform()

        xy = [(0,0),(0.2,0.4),(-0.3,-0.1)]
        for x,y in xy:
            px,py = XFORM.xy_to_pxpy(x,y)
            _x,_y = XFORM.pxpy_to_xy(px,py)
            self.assertEqual(x,_x)
            self.assertEqual(y,_y)

        m = XFORM.pixel_to_m(20)
        p = XFORM.m_to_pixel(m)
        self.assertEqual(p,20)

class TestFlyFlyPathModel(unittest.TestCase):

    def setUp(self):
        self._sdir = os.path.join(
                    roslib.packages.get_pkg_dir('strawlab_freeflight_experiments'),
                   'data','svgpaths'
        )

    def test_model_api(self):
        s = os.path.join(self._sdir,'lboxmed.svg')
        m = MovingPointSvgPath(s)
        m.start_move_from_ratio(0)
        for prop in ('polyline','svg_path_data','moving_pt','ratio'):
            o = getattr(m,prop)
            self.assertIsNotNone(o)

    def test_plot_helpers(self):
        for svg in ('infinity.svg', 'lboxmed.svg', 'plain.svg'):
            m = MovingPointSvgPath(os.path.join(self._sdir,svg))
            f = plt.figure()
            ax = f.add_subplot(1,2,1)
            ax.set_xlim(-1,1)
            ax.set_ylim(-1,1)
            pltpath.plot_xy(m,ax)
            ax = f.add_subplot(1,2,2)
            pltpath.plot_polygon(m,ax,fc='red',ec='black',alpha=0.7,label='original')
            try:
                pltpath.plot_polygon(m,ax,fc='none',ec='blue',scale=0.05,label='grow')
                pltpath.plot_polygon(m,ax,fc='none',ec='green',scale=-0.05,label='shrink')
            except ValueError:
                pass
            ax.set_xlim(-1,1)
            ax.set_ylim(-1,1)
            ax.legend()

        #plt.show()

    def test_hitmanager(self):
        #path crosses itself
        m = MovingPointSvgPath(os.path.join(self._sdir,'infinity.svg'))
        self.assertRaises(ValueError,HitManager,m,transform_to_world=True,validate=True)

        #path does not connect with itself
        m = MovingPointSvgPath(os.path.join(self._sdir,'plain.svg'))
        self.assertRaises(ValueError,HitManager,m,transform_to_world=True,validate=True)

        m = MovingPointSvgPath(os.path.join(self._sdir,'lboxmed.svg'))
        h = HitManager(m,transform_to_world=True,validate=True)
        self.assertTrue(h.contains(0.1,0.0))
        self.assertFalse(h.contains(100,0.0))

        x,y = h.points
        self.assertTrue(len(x) > 0)
        self.assertTrue(len(y) > 0)

        #missing file
        self.assertRaises(SvgError,MovingPointSvgPath,os.path.join(self._sdir,'MISSING.svg'))


if __name__ == '__main__':
    unittest.main()
