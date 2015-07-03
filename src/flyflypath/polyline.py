"""
Classes to represent beziers as segments of straight lines, and thus
be able to iterate along them and find the closest point along the
path to another point.

refs:
https://github.com/sporritt/jsBezier/blob/master/js/0.4/jsBezier-0.4.js
http://perrygeo.googlecode.com/svn/trunk/gis-bin/bezier_smooth.py
http://paulbourke.net/geometry/bezier/cubicbezier.html
http://www.lemoda.net/maths/bezier-length/index.html
http://www.antigrain.com/__code/src/agg_curves.cpp.html
"""

import numpy as np

from euclid import Point2, LineSegment2, Vector2

class ZeroLineSegment2:
    def __init__(self, pt=None):
        if pt is None:
            pt = Point2()
        self.p1 = pt
        self.p2 = pt
        self.v = Vector2()
        self.length = 0

    def __repr__(self):
        return "LineSegment2(%r to %r)" % (self.p1, self.p2)

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
        self.points = []
        self._lines = []

        #cull zero lenght line segments - it is easier to do it here then
        #when parsing the svg
        follow = 0
        lead = 1
        npts = len(points)
        while lead < npts:
            p1 = points[follow]
            p2 = points[lead]
            if p1 == p2:
                lead += 1
                continue
            else:
                pstart = points[follow]
                pend = points[lead]

                self._lines.append( LineSegment2(pstart, pend) )
                if not self.points:
                    self.points.append(pstart)
                self.points.append(pend)

                follow = lead
                lead += 1

        self._length = 0
        self._lengths = {}
        for l in self._lines:
            length = l.length
            self._lengths[l] = length
            self._length += length

    @property
    def lines(self):
        return self._lines

    @property
    def p1(self):
        """ first point """
        return self.points[0].copy()

    @property
    def length(self):
        return self._length

    @property
    def num_segments(self):
        return len(self._lines)

    def along(self, percent):
        target = self.length * percent
        this = next = 0
        for l in self._lines:
            ll = self._lengths[l]
            next = this + ll
            if next > target:
                break
            elif next == target:
                return l.p2
            else:
                this = next

        #l is the linesegment that contains the point, 'this' is the length
        #comsumed up to the start of the segment
        remain = target - this
        linesegmentlen = self._lengths[l]
        fraction = float(remain) / linesegmentlen
        return Point2(l.p1.x + fraction*l.v.x,
                      l.p1.y + fraction*l.v.y)
        
    def connect(self, point):
        closest = None
        closestl = None

        dist = np.inf
        for l in self._lines:
            if l.p1 == point or l.p2 == point:
                closest = point
                closestl = l
                closesttype = "vertex"
                break
            else:
                seg = l.connect(point)
                newdist = seg.length
                if newdist < dist:
                    #we are closer than before
                    closest = seg.p
                    closestl = l
                    closesttype = "segment"
                    dist = newdist

        #FIXME: iterate again... ARGH
        length = 0
        for l in self._lines:
            if l == closestl:
                length += (closest - l.p1).magnitude()
                break
            length += l.length

        return closest,length/self.length

    def __repr__(self):
        return 'PolyLine2(%s)' % ', '.join( repr(l) for l in self._lines )

class BezierSegment2:
    def __init__(self, p1, p2, p3, p4):
        """ start,control1,control2,end """
        #see 
        # [1] http://www.antigrain.com/research/adaptive_bezier/index.html
        # (use p1,p2,p3,p4 to be the same API as euler and [1])
        self.p1 = self.p = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4

    def evaluate(self, mu):
        mum1 = 1.0 - mu
        mum13 = mum1 * mum1 * mum1
        mu3 = mu * mu * mu

        x = mum13*self.p1.x + 3*mu*mum1*mum1*self.p2.x + 3*mu*mu*mum1*self.p3.x + mu3*self.p4.x
        y = mum13*self.p1.y + 3*mu*mum1*mum1*self.p2.y + 3*mu*mu*mum1*self.p3.y + mu3*self.p4.y

        return Point2(x,y)

    def to_points(self, num_steps=None):
        if num_steps:
            murange = np.linspace(0,1.0,num_steps)
        else:
            #make a reasonable guess according to [1]
            l1 = (self.p2-self.p1).magnitude()
            l2 = (self.p3-self.p2).magnitude()
            l3 = (self.p4-self.p3).magnitude()
            murange = np.linspace(0,1.0,int( (l1+l2+l3) * 0.25 ))
            
        return [self.evaluate(float(mu)) for mu in murange]
        
    @property
    def length(self):
        #see jsBezier for recursive approx, or just sum the length of the polyline2
        raise Exception("Not Supported")

    def __repr__(self):
        return 'BezierSegment2(%r -> %r (ctrl1:%r, ctrl2:%r))' % (self.p1,self.p4,self.p2,self.p3)


