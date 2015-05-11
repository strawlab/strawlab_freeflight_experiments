#!/usr/bin/env python
import os.path
import unittest

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')

import numpy as np

from flyflypath.euclid import Point2, LineSegment2, Vector2
from flyflypath.polyline import PolyBezier2, BezierSegment2, PolyLine2, ZeroLineSegment2
from flyflypath.transform import SVGTransform

class TestFlyFlyPath(unittest.TestCase):

    def setUp(self):
        self._sdir = os.path.join(
                    roslib.packages.get_pkg_dir('strawlab_freeflight_experiments'),
                   'data','svgpaths'
        )

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

if __name__ == '__main__':
    unittest.main()
