import os.path
import xml.dom.minidom

import numpy as np

import polyline
import euclid
import transform

import svgpath

from polyline import PolyLine2, BezierSegment2
from euclid import Point2

def represent_svg_path_as_polyline(pathdata, points_per_bezier):
    paths = svgpath.parse_path(pathdata)

    pts = []
    for path in paths:
        if isinstance(path, svgpath.Line):
            for pt in (path.start, path.end):
                pts.append(Point2(float(pt.real),float(pt.imag)))
        elif isinstance(path, (svgpath.CubicBezier, svgpath.QuadraticBezier)):
            for i in np.linspace(0,1.0,points_per_bezier):
                pt = path.point(i)
                pts.append(Point2(float(pt.real),float(pt.imag)))
        elif isinstance(path, svgpath.Arc):
            for i in np.linspace(0,1.0,20):
                pt = path.point(i)
                pts.append(Point2(float(pt.real),float(pt.imag)))

    return PolyLine2(*pts)

class SvgError(Exception):
    pass

class MultiplePathSvgError(SvgError):
    pass

class SvgPath(object):

    num_paths = 1

    def __init__(self, path, polyline=None, svg_path_data=None, npts=5):
        if svg_path_data is None:
            if not os.path.exists(path):
                raise SvgError("File Missing: %s" % path)
            #parse the SVG
            d = xml.dom.minidom.parse(open(path,'r'))
            paths = d.getElementsByTagName('path')
            if len(paths) != 1:
                raise MultiplePathSvgError("Only 1 path supported")
            svg_path_data = str(paths[0].getAttribute('d'))
        self._svg_path_data = svg_path_data

        if polyline is None:
            polyline = represent_svg_path_as_polyline(self._svg_path_data, npts)
        self._polyline = polyline

    @property
    def paths(self):
        return (self,)

    @property
    def polyline(self):
        return self._polyline

    @property
    def svg_path_data(self):
        return self._svg_path_data

    def get_approximation(self, npts):
        return SvgPath(path=None,
                       polyline=None,
                       svg_path_data=self._svg_path_data,
                       npts=npts)

    def get_hitmanager(self, xform, validate=True, scale=None):
        return SvgPathHitManager(self, xform, validate=validate, scale=scale)

    def point(self, pos, transform=None):
        pt = self._polyline.along(pos)
        if transform is not None:
            return transform.pxpy_to_xy(pt.x,pt.y)
        else:
            return pt.x,pt.y

    def get_points(self, transform=None):
        if transform is not None:
            tfunc = transform.pxpy_to_xy
        else:
            tfunc = lambda px,py: (px,py)
        return [tfunc(pt.x, pt.y) for pt in self._polyline.points]

    def connect_closest(self, p, px=None, py=None):
        """
        finds the closest point on the path to p

        p,px,py are in pixel coordinates. returns the
        line segment describing the connection between p and
        the path, and the ratio (0-1) describing how far
        along the path the closest point it
        """
        if px is not None and py is not None:
            p = euclid.Point2(px,py)
        closest,ratio = self._polyline.connect(p)
        try:
            seg = euclid.LineSegment2(p,closest)
        except AttributeError:
            seg = polyline.ZeroLineSegment2(closest)
        return seg,ratio

class MultipleSvgPath(object):

    def __init__(self, path, polyline=None, svg_path_data=None, npts=5):
        if not os.path.exists(path):
            raise SvgError("File Missing: %s" % path)
        #parse the SVG
        d = xml.dom.minidom.parse(open(path,'r'))
        paths = d.getElementsByTagName('path')

        self._filepath = path
        self._paths = tuple(SvgPath(path=None,polyline=None,svg_path_data=p.getAttribute('d'), transform=p.getAttribute('transform'), npts=npts) for p in paths)

    @property
    def num_paths(self):
        return len(self._paths)

    @property
    def paths(self):
        return self._paths

    def get_approximation(self, npts):
        return MultipleSvgPath(path=self._filepath,
                       polyline=None,
                       svg_path_data=None,
                       npts=npts)

    def get_hitmanager(self, xform, validate=True, scale=None):
        return MultipleSvgPathHitManager(self, xform, validate=validate, scale=scale)


class MovingPointSvgPath(SvgPath):

    def __init__(self, path, polyline=None):
        SvgPath.__init__(self,path,polyline)
        self._moving_pt = None
        self._ratio = 0.0

    @property
    def moving_pt(self):
        return self._moving_pt

    @property
    def ratio(self):
        return self._ratio

    def start_move_from_ratio(self, ratio):
        """ set the ratio and point to this """
        self._ratio = ratio
        self._moving_pt = self._polyline.along(self._ratio)
        return self._ratio, self._moving_pt

    def advance_point(self, delta, wrap=False):
        return self.move_point(self._ratio + delta, wrap)

    def move_point(self, ratio, wrap=False):
        """ the amount along the path [0..1], 0=start, 1=end """
        ratio = min(1.0,max(0.0,ratio))
        if wrap and (ratio == 1.0):
            ratio = 0.0
        self._ratio = ratio
        self._moving_pt = self._polyline.along(self._ratio)
        return self._ratio, self._moving_pt

    def connect_to_moving_point(self, p, px=None, py=None):
        """
        finds the vector connecting p to the moving point

        p,px,py are in pixel coordinates.
        """
        if px is not None and py is not None:
            p = euclid.Point2(px,py)
        try:
            seg = euclid.LineSegment2(p,self._moving_pt)
        except AttributeError:
            seg = polyline.ZeroLineSegment2(p)
        return seg

class PathError(Exception):
    pass

class OpenPathError(PathError):
    pass

class InvalidPathError(PathError):
    pass

class SvgPathHitManager(object):

    num_paths = 1

    def __init__(self, model, transform, validate=True, scale=None):

        if model.num_paths > 1:
            raise PathError("HitManager only supports single paths")

        import shapely.geometry as sg
        from shapely.geometry.polygon import LinearRing, LineString
        from shapely.validation import explain_validity

        self._polyline = model
        self._t = transform

        #keep the model points in pixels
        coords = model.get_points(transform=None)

        if validate:
            #sg.LinearRing and sg.Polygon are automatically closed, so
            #passing a non-closed path could give counterintuative results.

            p0,pn = coords[0], coords[-1]
            if not np.allclose(p0,pn):
                raise OpenPathError('Invalid model: path is not closed')

            #also, AFAICT overlapping paths can be problematic for testing if
            #a polygon contains a point - although the shapely docs are not clear
            #on if this is a problem or not.......
            #
            #geometry should be able to be described by a linear ring - i.e.
            #"A LinearRing may not cross itself, and may not touch itself at a single point"
            #so infinity paths could be incorrect
            lr = LinearRing(coords)

            if not lr.is_valid:
                #there are some numerical issues with how inkscape draws
                #circles/ellipses as 2 or 4 beziers. Round those errors away... hopefully
                coords = [(round(c[0],5),round(c[1],5)) for c in coords]
                lr = LinearRing(coords)

            if not lr.is_valid:
                raise InvalidPathError('Invalid model: %s' % explain_validity(lr))

            self._poly = sg.Polygon(lr)
        else:
            self._poly = sg.Polygon(coords)

        #scale/buffer the polygon out or in
        if scale is not None:
            if not self._poly.is_valid:
                raise ValueError('Could not scale invalid polygon')
            self._poly = self._poly.buffer(scale)
            if hasattr(self._poly,'geoms'):
                raise ValueError('Sclaing split polygon into multiple polygons')

        self._pt = sg.Point #sorry lazy import

    @property
    def points(self):
        return self._poly.exterior.xy

    def contains_px(self, px, py):
        return self._poly.contains(self._pt(px,py))

    def contains_m(self, x, y):
        px,py = self._t.xy_to_pxpy(x,y)
        return self.contains_px(px,py)

    def distance_to_closest_point_px(self, px, py):
        try:
            seg,ratio = self._polyline.connect_closest(None,px=float(px),py=float(py))
            return seg.length
        except Exception as e:
            print e
            return np.nan

    def distance_to_closest_point_m(self, x, y):
        px,py = self._t.xy_to_pxpy(x,y)
        return self.distance_to_closest_point_px(px,py)

class MultipleSvgPathHitManager(object):

    def __init__(self, model, transform, validate=True, scale=None):
        self._hm = tuple(SvgPathHitManager(p,transform,validate=validate,scale=scale) for p in model.paths)

    @property
    def num_paths(self):
        return len(self._hm)

    @property
    def points(self):
        return [hm.points for hm in self._hm]

    def contains_px(self, px, py):
        return any(hm.contains_px(px,py) for hm in self._hm)

    def contains_m(self, x, y):
        return any(hm.contains_m(x,y) for hm in self._hm)

    def distance_to_closest_point_px(self, px, py):
        return [hm.disance_to_closest_point_px(px,py) for hm in self._hm]

    def distance_to_closest_point_m(self, x, y):
        return [hm.disance_to_closest_point_m(x,y) for hm in self._hm]

