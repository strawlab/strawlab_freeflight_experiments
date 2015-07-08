import math

import cairo
import numpy as np
from gi.repository import Gtk, Gdk

import euclid

import svgpath

def draw_on_context(cr, pathdata):
    for path in svgpath.parse_path(pathdata):
        cr.move_to(path.start.real,path.start.imag)
        if isinstance(path, svgpath.Line):
            cr.line_to(path.end.real,path.end.imag)
        elif isinstance(path, svgpath.CubicBezier):
            cr.curve_to(path.control1.real,path.control1.imag,
                        path.control2.real,path.control2.imag,
                        path.end.real,path.end.imag)
        elif isinstance(path, svgpath.Arc):
            #the cairo api for arcs is mental
            #elliptical arcs can be approximated as beziers
            #http://www.spaceroots.org/documents/ellipse/elliptical-arc.pdf
            #http://pomax.github.io/bezierinfo/#circles_cubic
            #
            #circles can be approximated as 4 beziers according to
            #http://spencermortensen.com/articles/bezier-circle/
            #
            #do none of these things and just approximate the arc with points
            for i in np.linspace(0,1.0,50):
                pt = path.point(i)
                cr.line_to(pt.real,pt.imag)
        else:
            raise Exception("Not Supported")


class ViewWidget(Gtk.DrawingArea):
    def __init__(self):
        super(ViewWidget, self).__init__()
        self.connect('draw', self._on_draw_event)
        self.connect('configure-event', self._on_configure_event)
        self._surface = None
        self._need_bg_redraw = False

    def redraw(self):
        if self._need_bg_redraw:
            cr = cairo.Context(self._surface)
            self.redraw_background(cr)
            self.add_to_background(cr)
            self._need_bg_redraw = False
        self.queue_draw()
        return True

    def draw_background(self):
        self._need_bg_redraw = True
       
    def redraw_background(self, cr):
        cr.set_source_rgb(1, 1, 1)
        cr.paint()

    def add_to_background(self, cr):
        """ override this for a chance to draw additonal stuff on the background """
        pass

    def _on_draw_event(self, widget, cr):
        cr.set_source_surface(self._surface, 0.0, 0.0)
        cr.paint()

        vecs,pts,polys = self.get_vec_and_points()

        for vec,rgb in vecs:
            cr.set_source_rgb (*rgb)
            cr.set_line_width (1)
            cr.move_to(vec.p1.x,vec.p1.y)
            cr.line_to(vec.p2.x,vec.p2.y)
            cr.stroke()

        offset = 12
        for pt,rgb in pts:
            if pt is not None:
                cr.set_source_rgb (*rgb)
                cr.move_to(pt.x, pt.y)
                cr.arc(pt.x, pt.y, 2, 0, 2.0 * math.pi)
                cr.fill()
                cr.move_to(5,offset)
                try:
                    #the reason I don't show x: and y: here is because at this
                    #stage they are in pixels (and z is in m). I think it would
                    #be a layring violation to have the transform in here, however
                    #it is undoubtedly ugly to have different units for x,y and z.
                    #
                    #C'est la vie I guess.
                    cr.show_text("z: %.2f" % pt.z)
                    cr.stroke()
                except AttributeError:
                    pass
                offset += 12


        for poly,rgb in polys:
            if poly:
                #draw the approximation
                cr.set_source_rgb (*rgb)
                cr.set_line_width (0.5)
                cr.move_to(poly[0].x,poly[0].y)
                for i in range(1,len(poly)):
                    cr.line_to(poly[i].x,poly[i].y)
                cr.close_path()
                cr.stroke()

        annots = self.get_annotations()
        for pt,rgb,txt in annots:
            cr.set_source_rgb (*rgb)
            cr.move_to(pt.x, pt.y)
            cr.show_text(txt)
            cr.stroke()

    def _on_configure_event(self, widget, event):
        allocation = self.get_allocation()
        self._surface = self.get_window().create_similar_surface(
                                            cairo.CONTENT_COLOR,
                                            allocation.width,
                                            allocation.height)
        self.draw_background()
        self.redraw()

    def get_vec_and_points(self):
        """
        returns a 3-tuple
            [(euclid.LineSegment,(r,g,b)),]
            [(euclid.Point2,(r,g,b)),]
            [([euclid.Point2,],(r,g,b)),]
        """
        raise NotImplementedError

    def get_annotations(self):
        """
        returns a list of 3-tuples
            [(euclid.Point2,(r,g,b),"txt"),...]
        """
        return []

class SvgPathWidget(ViewWidget):

    def __init__(self, model, transform):
        super(SvgPathWidget, self).__init__()
        self.set_size_request(*transform.size_px)
        self._model = model

    @property
    def model(self):
        return self._model

    def redraw_background(self, cr):
        ViewWidget.redraw_background(self,cr)

        if not self._model:
            return

        cr.set_source_rgb (0, 0, 0)
        cr.set_line_width (1)

        draw_on_context(cr, self._model.svg_path_data)
        cr.stroke()

        #draw the approximation
        polyline = self._model.polyline
        cr.set_source_rgb (0, 0, 1)
        cr.set_line_width (1)
        cr.set_dash([1.0])
        cr.move_to(polyline.points[0].x,polyline.points[0].y)
        for i in range(1,len(polyline.points)):
            cr.line_to(polyline.points[i].x,polyline.points[i].y)
        cr.stroke()


