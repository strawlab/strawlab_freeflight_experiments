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
from flyflypath.model import MovingPointSvgPath, SvgPath, SvgPathHitManager, SvgError, InvalidPathError, OpenPathError, MultipleSvgPath, MultiplePathSvgError, MultipleSvgPathHitManager

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
        self._tdir = os.path.join(
                    roslib.packages.get_pkg_dir('strawlab_freeflight_experiments'),
                   'test','data'
        )

    def test_model_api(self):
        s = os.path.join(self._sdir,'lboxmed.svg')
        m = MovingPointSvgPath(s)
        m.start_move_from_ratio(0)
        for prop in ('polyline','svg_path_data','moving_pt','ratio'):
            o = getattr(m,prop)
            self.assertIsNotNone(o)

    def test_plot_helpers(self):
        transform = SVGTransform()

        for svg in ('infinity.svg', 'lboxmed.svg', 'plain.svg'):
            m = MovingPointSvgPath(os.path.join(self._sdir,svg))
            f = plt.figure()
            ax = f.add_subplot(1,2,1)
            ax.set_xlim(-1,1)
            ax.set_ylim(-1,1)
            pltpath.plot_xy(m,transform,ax)
            ax = f.add_subplot(1,2,2)
            pltpath.plot_polygon(m,transform,ax,fc='red',ec='black',alpha=0.7,label='original')
            try:
                pltpath.plot_polygon(m,transform,ax,fc='none',ec='blue',scale=0.05,label='grow')
                pltpath.plot_polygon(m,transform,ax,fc='none',ec='green',scale=-0.05,label='shrink')
            except ValueError:
                pass
            ax.set_xlim(-1,1)
            ax.set_ylim(-1,1)
            ax.legend()

        #plt.show()

    def test_svgpathhitmanager(self):
        transform = SVGTransform()

        #path crosses itself
        m = SvgPath(os.path.join(self._sdir,'infinity.svg'))
        self.assertRaises(InvalidPathError,SvgPathHitManager,m,transform,validate=True)

        #path does not connect with itself
        m = SvgPath(os.path.join(self._sdir,'plain.svg'))
        self.assertRaises(OpenPathError,SvgPathHitManager,m,transform,validate=True)

        m = SvgPath(os.path.join(self._sdir,'lboxmed.svg'))
        h = SvgPathHitManager(m,transform,validate=True)
        self.assertTrue(h.contains_m(0.1,0.0))
        self.assertTrue(h.contains_px(250,250))
        self.assertFalse(h.contains_m(100,0.0))

        x,y = h.points
        self.assertTrue(len(x) > 0)
        self.assertTrue(len(y) > 0)

        #missing file
        self.assertRaises(SvgError,MovingPointSvgPath,os.path.join(self._sdir,'MISSING.svg'))

    def test_path(self):
        m = SvgPath(os.path.join(self._tdir,'bezier.svg'))
        self.assertEqual(m.num_paths, 1)

        transform = SVGTransform()
        hm = m.get_hitmanager(transform)
        self.assertTrue(isinstance(hm,SvgPathHitManager))

        m2 = m.get_approximation(400)
        self.assertTrue(isinstance(m2,SvgPath))

    def test_bezier(self):
        m = SvgPath(os.path.join(self._tdir,'bezier.svg'))
        self.assertEqual(len(m.get_points()),57)

        ###test hitmanager
        transform = SVGTransform()
        hm = m.get_hitmanager(transform)

        self.assertEqual(hm.distance_to_closest_point_m(0.0,0.0), 0.032596266707763916)
        self.assertEqual(hm.distance_to_closest_point_px(0.0,0.0), 204.75459496191291)

    def test_simple_path(self):
        def _check_start_path(_x,_y):
            #from svg path data, first M command (start of path)
            self.assertEqual(_x,95.878886)
            self.assertEqual(_y,173.41255)

        def _check_ls_to_start_of_path(_ls):
            self.assertEqual(_ls.p1.x, 0.0)
            self.assertEqual(_ls.p1.y, 0.0)

            x,y = _ls.p2.x, _ls.p2.y
            _check_start_path(x,y)

        for npts in (5,10,19):
            m = SvgPath(os.path.join(self._tdir,'bezier_simple.svg'), npts=npts)
            self.assertEqual(len(m.get_points()), npts)

        self.assertEqual(len(m.get_approximation(100).get_points()),100)

        m = SvgPath(os.path.join(self._tdir,'bezier_simple.svg'), npts=5)
        self.assertEqual(m.point(0.5), (213.1052380714953, 237.21054620573256))

        ls,ratio = m.connect_closest(None,px=0,py=0)
        self.assertEqual(ratio,0.0)
        _check_ls_to_start_of_path(ls)

        ###test moving point path
        m2 = MovingPointSvgPath(os.path.join(self._tdir,'bezier_simple.svg'))
        ratio,p = m2.start_move_from_ratio(0)
        self.assertEqual(ratio,0.0)

        ls = m2.connect_to_moving_point(None, px=0, py=0)
        _check_ls_to_start_of_path(ls)

        ratio,p = m2.advance_point(0.0)
        self.assertEqual(ratio,0.0)
        _check_start_path(p.x,p.y)
        ls = m2.connect_to_moving_point(None, px=0, py=0)
        _check_ls_to_start_of_path(ls)

        ratio,p = m2.move_point(0)
        self.assertEqual(ratio,0.0)
        _check_start_path(p.x,p.y)
        ls = m2.connect_to_moving_point(None, px=0, py=0)
        _check_ls_to_start_of_path(ls)

        ratio,p = m2.move_point(1.0,wrap=True)
        self.assertEqual(ratio,0.0)
        _check_start_path(p.x,p.y)
        ls = m2.connect_to_moving_point(None, px=0, py=0)
        _check_ls_to_start_of_path(ls)

    def test_multipath(self):
        self.assertRaises(MultiplePathSvgError,SvgPath,os.path.join(self._tdir,'multiple.svg'))

        m = MultipleSvgPath(os.path.join(self._tdir,'multiple.svg'))
        self.assertEqual(m.num_paths, 4)

        transform = SVGTransform()
        hm = m.get_hitmanager(transform)
        self.assertTrue(isinstance(hm,MultipleSvgPathHitManager))

        m2 = m.get_approximation(400)
        self.assertTrue(isinstance(m2,MultipleSvgPath))


if __name__ == '__main__':
    unittest.main()
