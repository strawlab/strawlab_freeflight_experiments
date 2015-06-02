import os.path
import xml.dom.minidom

import numpy as np

import svg
import polyline
import euclid
import transform

APPROX_BEZIER_WITH_N_SEGMENTS = 5

class SvgError(Exception):
    pass

class MovingPointSvgPath:
    def __init__(self, path):
        if not os.path.exists(path):
            raise SvgError("File Missing: %s" % path)
        self._svgpath = path
        #parse the SVG
        d = xml.dom.minidom.parse(open(path,'r'))
        paths = d.getElementsByTagName('path')
        if len(paths) != 1:
            raise SvgError("Only 1 path supported")
        pathdata = str(paths[0].getAttribute('d'))

        self._svgiter = svg.PathIterator(pathdata)
        self._model = polyline.polyline_from_svg_path(self._svgiter, APPROX_BEZIER_WITH_N_SEGMENTS)
        self._moving_pt = None
        self._ratio = 0.0

    @property
    def polyline(self):
        return self._model

    @property
    def svgiter(self):
        return self._svgiter

    @property
    def svgpath(self):
        return self._svgpath

    @property
    def moving_pt(self):
        return self._moving_pt

    @property
    def ratio(self):
        return self._ratio

    def get_points(self, transform_to_world):
        if transform_to_world:
            t = transform.SVGTransform()
            tfunc = t.pxpy_to_xy
        else:
            tfunc = lambda px,py: (px,py)
        return [tfunc(pt.x, pt.y) for pt in self._model.points]

    def start_move_from_ratio(self, ratio):
        """ set the ratio and point to this """
        self._ratio = ratio
        self._moving_pt = self._model.along(self._ratio)
        return self._ratio, self._moving_pt

    def advance_point(self, delta, wrap=False):
        return self.move_point(self._ratio + delta, wrap)

    def move_point(self, ratio, wrap=False):
        """ the amount along the path [0..1], 0=start, 1=end """
        ratio = min(1.0,max(0.0,ratio))
        if wrap and (ratio == 1.0):
            ratio = 0.0
        self._ratio = ratio
        self._moving_pt = self._model.along(self._ratio)
        return self._ratio, self._moving_pt

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
        closest,ratio = self._model.connect(p)
        try:
            seg = euclid.LineSegment2(p,closest)
        except AttributeError:
            seg = polyline.ZeroLineSegment2(closest)
        return seg,ratio

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

class HitManager:
    def __init__(self, model, transform_to_world, validate=True, scale=None):

        import shapely.geometry as sg
        from shapely.geometry.polygon import LinearRing, LineString
        from shapely.validation import explain_validity

        coords = model.get_points(transform_to_world)

        if validate:
            #sg.LinearRing and sg.Polygon are automatically closed, so
            #passing a non-closed path could give counterintuative results.

            p0,pn = coords[0], coords[-1]
            if not np.allclose(p0,pn):
                raise ValueError('Invalid model: path is not closed')

            #also, AFAICT overlapping paths can be problematic for testing if
            #a polygon contains a point - although the shapely docs are not clear
            #on if this is a problem or not.......
            #
            #geometry should be able to be described by a linear ring - i.e.
            #"A LinearRing may not cross itself, and may not touch itself at a single point"
            #so infinity paths could be incorrect
            lr = LinearRing(coords)
            if not lr.is_valid:
                raise ValueError('Invalid model: %s' % explain_validity(lr))

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

    def contains(self, x, y):
        return self._poly.contains(self._pt(x,y))

    

