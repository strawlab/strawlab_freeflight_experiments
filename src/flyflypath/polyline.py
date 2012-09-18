import numpy as np
import collections

from euclid import Point2, Vector2, LineSegment2

class ClosestPoint(object):
    __slots__= "point", "vector", "type"

    def __init__(self, point=None, vector=None, type=None):
        self.point = point
        self.vector = vector
        self.type = type
    
    def __repr__(self):
        return "ClosestPoint(%r, %r, %s)" % (self.point, self.vector, self.type)


class PolyLine2:
    def __init__(self, *points):
        self._lines = [ LineSegment2(points[i], points[i+1]) \
                        for i in range(len(points)-1)]

    @property
    def length(self):
        return sum( l.length for l in self._lines )
        
    def connect(self, point):
        closest = ClosestPoint()

        dist = np.inf
        for l in self._lines:
            if l.p1 == point or l.p2 == point:
                closest.point = point
                closest.type = "vertex"
                break
            else:
                seg = l.connect(point)
                newdist = seg.length
                if newdist < dist:
                    closest.point = seg.p
                    closest.vector = seg.v
                    closest.type = "segment"
                    dist = newdist

        return closest

    def __repr__(self):
        return 'PolyLine2(%s)' % ', '.join( repr(p) for p in self._points )

class Bezier2:
    def __init__(self, p0, p1, p2, p3):
        pass

def from_svg_path(self, text):
    pass

if __name__ == "__main__":
    p0 = Point2(1,1)
    p1 = Point2(1,2)
    p2 = Point2(2,2)
    p4 = Point2(3,3)

    assert (p0-p1).magnitude() == 1

    l0 = LineSegment2(p0,p1)
    l1 = LineSegment2(p1,p2)

    assert l0.length == 1
    assert l1.length == 1
    assert LineSegment2(p0,p2).length == np.sqrt(2)
    assert (p2-p0).magnitude_squared() == 2
    
    tp0 = Point2(0.3,0.4)
    tp1 = Point2(1.2,2.2)
    tp2 = Point2(2,2)
    tp3 = Point2(2.8,2.2)
    tp4 = Point2(30,40)

    p = PolyLine2(p0,p1,p2,p4)

    print p.connect(tp0)
    print p.connect(tp1)
    print p.connect(tp2)
    print p.connect(tp3)
    print p.connect(tp4)
    

    #print p.connect(Point2(1.8,2.2))
    #print p.connect(Point2(1,2))
    #print p.connect()
    
    
#https://github.com/sporritt/jsBezier/blob/master/js/0.4/jsBezier-0.4.js
#http://perrygeo.googlecode.com/svn/trunk/gis-bin/bezier_smooth.py
#http://paulbourke.net/geometry/bezier/cubicbezier.html
#http://www.lemoda.net/maths/bezier-length/index.html
#http://www.antigrain.com/__code/src/agg_curves.cpp.html

    
