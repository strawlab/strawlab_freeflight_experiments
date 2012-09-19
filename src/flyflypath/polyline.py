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

    @property
    def length(self):
        return sum( l.length for l in self._lines )

    @property
    def num_segments(self):
        return len(self._lines)
        
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
        
    def to_polyline(self, num_steps=None):
        return PolyLine2( *self.to_points(num_steps) )

    @property
    def length(self):
        #see jsBezier for recursive approx, or just sum the length of the polyline2
        raise Exception("Not Supported")

    def __repr__(self):
        return 'BezierSegment2(%r -> %r (ctrl1:%r, ctrl2:%r))' % (self.p1,self.p4,self.p2,self.p3)

class PolyBezier2:
    def __init__(self, *points):
        self._lines = [ BezierSegment2(points[i+0], points[i+1], points[i+2], points[i+3]) \
                        for i in range(0,len(points)-3,4)]

    def __repr__(self):
        return 'PolyBezier2(%s)' % ', '.join( repr(l) for l in self._lines )

def polyline_from_svg_path(path_iterator, points_per_bezier=10):
    pts = []
    last = Point2(0,0)
    for path_element in path_iterator:
        command, c = path_element
        if command == "M" or command == "L":
            for i in range(0,len(c)-1,2):
                pts.append( Point2(c[i+0],c[i+1]) )
                last = pts[-1]
        elif command == "m" or command == "l":
            for i in range(0,len(c)-1,2):
                #now we are relative movement wrt last point
                lastx = last.x
                lasty = last.y
                pts.append( Point2(c[i+0]+lastx,c[i+1]+lasty) )
                last = pts[-1]
        elif command == "C":
            for i in range(0,len(c)-5,6):
                b = BezierSegment2(
                        last.copy(),
                        Point2(c[i+0],c[i+1]),
                        Point2(c[i+2],c[i+3]),
                        Point2(c[i+4],c[i+5]))
                last = b.p4
                pts.extend( b.to_points(points_per_bezier) )
        elif command == "c":
            for i in range(0,len(c)-5,6):
                lastx = last.x
                lasty = last.y
                b = BezierSegment2(
                        last.copy(),
                        Point2(c[i+0]+lastx,c[i+1]+lasty),
                        Point2(c[i+2]+lastx,c[i+3]+lasty),
                        Point2(c[i+4]+lastx,c[i+5]+lasty))
                last = b.p4
                pts.extend( b.to_points(points_per_bezier) )
        else:
            raise Exception("Command %s not supported" % command)

    return PolyLine2(*pts)

if __name__ == "__main__":

    p0 = Point2(1,1)
    p1 = Point2(1,2)
    p1b = Point2(1,2)
    p2 = Point2(2,2)
    p4 = Point2(3,3)
    p4b = Point2(3,3)
    
    polya = PolyLine2(p0,p1,p1b,p2,p4,p4b)
    polyb = PolyLine2(p0,p1,p2,p4)
    
    assert polya.length == polyb.length
    assert polya.num_segments == polyb.num_segments
    
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
    
    b = BezierSegment2(
            Point2(50.507627,77.756235),
            Point2(160.61426,97.959286),
            Point2(208.09142,156.54813),
            Point2(153.54319,215.13698))
    print b
    print b.to_polyline(4)

#https://github.com/sporritt/jsBezier/blob/master/js/0.4/jsBezier-0.4.js
#http://perrygeo.googlecode.com/svn/trunk/gis-bin/bezier_smooth.py
#http://paulbourke.net/geometry/bezier/cubicbezier.html
#http://www.lemoda.net/maths/bezier-length/index.html
#http://www.antigrain.com/__code/src/agg_curves.cpp.html

    
