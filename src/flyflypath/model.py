import os.path
import xml.dom.minidom

import svg
import polyline
import euclid

APPROX_BEZIER_WITH_N_SEGMENTS = 5

class MovingPointSvgPath:
    def __init__(self, path):
        assert os.path.exists(path)
        self._svgpath = path
        #parse the SVG
        d = xml.dom.minidom.parse(open(path,'r'))
        paths = d.getElementsByTagName('path')
        assert len(paths) == 1
        pathdata = str(paths[0].getAttribute('d'))

        self._svgiter = svg.PathIterator(pathdata)
        self._model = polyline.polyline_from_svg_path(self._svgiter, APPROX_BEZIER_WITH_N_SEGMENTS)
        self._moving_pt = None

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

    def move_point(self, ratio):
        """ the amount along the path [0..1], 0=start, 1=end """
        ratio = min(1.0,max(0.0,ratio))
        self._moving_pt = self._model.along(ratio)
        return ratio

    def connect_closest(self, p, px=None, py=None):
        if px is not None and py is not None:
            p = euclid.Point2(px,py)
        closest = self._model.connect(p)
        return euclid.LineSegment2(p,closest)

    def connect_to_moving_point(self, p, px=None, py=None):
        if px is not None and py is not None:
            p = euclid.Point2(px,py)
        return euclid.LineSegment2(p,self._moving_pt)



