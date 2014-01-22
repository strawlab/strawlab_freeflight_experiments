
import svg
import math

import cairo
from gi.repository import Gtk, Gdk

import euclid

class SvgPathWidget(Gtk.DrawingArea):
    def __init__(self, model):
        Gtk.DrawingArea.__init__(self)
        self.set_size_request(500,500)
        self.connect('draw', self._on_draw_event)
        self.connect('configure-event', self._on_configure_event)

        self._model = model
        self._surface = None
        self._need_bg_redraw = False

    def redraw(self):
        if self._need_bg_redraw:
            self._draw_background()
            self._need_bg_redraw = False
        self.queue_draw()
        return True

    def draw_background(self):
        self._need_bg_redraw = True
       
    def _draw_background(self):
        cr = cairo.Context(self._surface)
        cr.set_source_rgb(1, 1, 1)
        cr.paint()

        if not self._model:
            return

        polyline = self._model.polyline
        svgiter = self._model.svgiter

        cr.set_source_rgb (0, 0, 0)
        cr.set_line_width (1)
        for path_element in svgiter:
            svg.draw_on_context(cr, path_element)
        cr.stroke()
        
        #draw the approximation
        cr.set_source_rgb (0, 0, 1)
        cr.set_line_width (0.3)
        cr.move_to(polyline.points[0].x,polyline.points[0].y)
        for i in range(1,len(polyline.points)):
            cr.line_to(polyline.points[i].x,polyline.points[i].y)
        cr.stroke()

        self.add_to_background(cr)

    def add_to_background(self, cr):
        """ override this for a chance to draw additonal stuff on the background """
        pass

    def _on_draw_event(self, widget, cr):
        cr.set_source_surface(self._surface, 0.0, 0.0)
        cr.paint()

        vecs,pts,poly = self.get_vec_and_points()

        for vec in vecs:
            cr.set_source_rgb (1, 0, 0)
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
                    cr.show_text("z: %.2f" % pt.z)
                    cr.stroke()
                except AttributeError:
                    pass
                offset += 12


        if poly:
            #draw the approximation
            cr.set_source_rgb (0, 0, 1)
            cr.set_line_width (0.3)
            cr.move_to(poly[0].x,poly[0].y)
            for i in range(1,len(poly)):
                cr.line_to(poly[i].x,poly[i].y)
            cr.close_path()
            cr.stroke()

    def _on_configure_event(self, widget, event):
        allocation = self.get_allocation()
        self._surface = self.get_window().create_similar_surface(
                                            cairo.CONTENT_COLOR,
                                            allocation.width,
                                            allocation.height)

        self._draw_background()

    def get_vec_and_points(self):
        """
        returns a 3-tuple
            euclid.LineSegment
            [(euclid.Point2,(r,g,b)),]
            [euclid.Point2,]
        """
        raise NotImplementedError


