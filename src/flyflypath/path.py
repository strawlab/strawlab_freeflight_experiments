import os.path
import xml.dom.minidom

import svg
import polyline
import euclid
import math

APPROX_BEZIER_WITH_N_SEGMENTS = 5

class SvgPath:
    def __init__(self, path):
        assert os.path.exists(path)
        #parse the SVG
        d = xml.dom.minidom.parse(open(path,'r'))
        paths = d.getElementsByTagName('path')
        assert len(paths) == 1
        pathdata = str(paths[0].getAttribute('d'))

        self._svgiter = svg.PathIterator(pathdata)
        self._model = polyline.polyline_from_svg_path(self._svgiter, APPROX_BEZIER_WITH_N_SEGMENTS)

    @property
    def polyline(self):
        return self._model

    @property
    def svgiter(self):
        return self._svgiter
