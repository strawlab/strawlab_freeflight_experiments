import re
import collections
import sys
import xml.dom.minidom

import numpy as np
import matplotlib.pyplot as plt

import model
import transform
from svgpath import parse_path

from polyline import PolyLine2, BezierSegment2
from euclid import Point2
from mplview import plot_xy, plot_polygon

APPROX_BEZIER_WITH_N_SEGMENTS = 5

def load_svg_path(path):
    d = xml.dom.minidom.parse(open(path,'r'))
    paths = d.getElementsByTagName('path')
    if len(paths) != 1:
        raise model.SvgError("Only 1 path supported")
    return str(paths[0].getAttribute('d'))

class PathIterator(object):
    PATH_IDENTIFIERS = r'[MLHVCSQTAmlhvcsqa]'
    NUMBERS = r'[0-9.-^A-z]'
    SEPERATORS = r'\s|\,'
    PATH_END = r'[Zz]'

    def __init__(self, path):
        self._parseable = path.translate(None, '\t\f')
        self._parseable = self._parseable.replace('\n', ' ')
        #strip_newlines
        self._parseable = re.sub(r'([A-Za-z])([0-9]|\-)', self._insert, self._parseable)
        #add_space
        self._parseable = self._parseable.replace(',', ' ')
        #replace_commas
        self._parseable = re.sub(r'\s+', ' ', self._parseable) # replace any double space with a single space
        #strip_extra_space
        self._tokens = re.split(' ', self._parseable)
        self._map = self._produce_map(self._tokens)
        self._mm = self._process(self._map)

    def _produce_map(self, tkns):
        m = collections.OrderedDict()
        i = 0
        while i < len(tkns):
            if re.match(self.PATH_IDENTIFIERS, tkns[i]):
                m[i] = tkns[i]
            elif re.match(self.PATH_END, tkns[i]):
                m[i] = tkns[i]
            else:
                pass
            i += 1
        return m.items()

    def _process(self, _map):
        mm = []
        l = len(_map)
        for e in range(l):
            try:
                element = _map[e]
                future = _map[e + 1]
                ident = element[1]
                start = element[0] + 1
                end = future[0]
                nbrs = self._tokens[start:end]
            except:
                element = _map[e]
                ident = element[1]
                start = element[0] + 1
                end = len(self._tokens)
                nbrs = self._tokens[start:end]
            finally:
                numbers = []
                for number in nbrs:
                    numbers.append(float(number))
                mm.append((ident, numbers))
        return mm

    def __iter__(self):
        return iter(self._mm)

    def _insert(self, match_obj):
        group = match_obj.group()
        return '{} {}'.format(group[0], group[1])

def represent_svg_path_as_polyline(pathdata, points_per_bezier):
    path_iterator = PathIterator(pathdata)

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

        elif command in ("z","Z"):
            #FIXME: The spec says join to the start of the last open path, I think we
            #only support 1 path, so this is the first point, not last.
            pts.append( pts[0] )
        else:
            raise Exception("Command %s not supported" % command)

    return PolyLine2(*pts)

if __name__ == "__main__":

    f = plt.figure("compare key points",figsize=(8,6))
    ax = f.add_subplot(1,1,1)

    t = transform.SVGTransform()

    #old parsing code
    try:
        mod = model.SvgPath(sys.argv[1],
                        polyline=represent_svg_path_as_polyline(load_svg_path(sys.argv[1]),APPROX_BEZIER_WITH_N_SEGMENTS))
        pts = mod.get_points(transform=t)
        x,y = np.array(pts).T
        ax.plot(x, y, 'k-', lw=0.2, label='old parsing code')

        for p in np.linspace(0,1.0,20):
            ptx,pty = mod.point(p, t)
            ax.plot(ptx, pty, 'k<')
    except Exception as e:
        print e

    #new parsing code
    try:
        mod2 = model.SvgPath(sys.argv[1])
        pts2 = mod2.get_points(transform=t)
        x2,y2 = np.array(pts2).T

        ax.plot(x2, y2, 'r-', lw=0.2, label='new parsing code')

        for p in np.linspace(0,1.0,20):
            ptx,pty = mod2.point(p, t)
            ax.plot(ptx, pty, 'r>')
        ax.legend()
    except model.MultiplePathSvgError:
        mod2 = model.MultipleSvgPath(sys.argv[1])

    #mpl helpers
    f = plt.figure("mpl view helper",figsize=(17,6))
    ax = f.add_subplot(1,2,1)
    plot_xy(mod2, t, ax)
    ax.set_xlim(-0.5,0.5)
    ax.set_ylim(-0.5,0.5)
    ax2 = f.add_subplot(1,2,2)
    plot_polygon(mod2, t, ax2)
    ax2.set_xlim(-0.5,0.5)
    ax2.set_ylim(-0.5,0.5)

    plt.show()
