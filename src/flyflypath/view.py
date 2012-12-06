
import svg
import math

import cairo
from gi.repository import Gtk, Gdk

import euclid

class SvgPathWidget(Gtk.DrawingArea):
    def __init__(self, model, to_pt=False):
        Gtk.DrawingArea.__init__(self)
        self.set_size_request(500,500)
        self.add_events(
                Gdk.EventMask.POINTER_MOTION_HINT_MASK | \
                Gdk.EventMask.POINTER_MOTION_MASK)
        self.connect('motion-notify-event', self._on_motion_notify_event)
        self.connect('draw', self._on_draw_event)
        self.connect('configure-event', self._on_configure_event)

        self._model = model
        self._snap_to_moving_pt = to_pt

        self._surface = None
        self._mousex = self._mousey = None
       
    def _draw_background(self):
        if not self._model:
            return

        polyline = self._model.polyline
        svgiter = self._model.svgiter

        cr = cairo.Context(self._surface)
        cr.set_source_rgb(1, 1, 1)
        cr.paint()

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

    def _on_motion_notify_event(self, da, event):
        (window, self._mousex, self._mousey, state) = event.window.get_pointer()
        self.queue_draw()
        return True

    def _on_draw_event(self, widget, cr):
        cr.set_source_surface(self._surface, 0.0, 0.0)
        cr.paint()

        vec,src_pt,trg_pt = self.get_vec_and_points()
        
        if vec is not None:
            cr.set_source_rgb (1, 0, 0)
            cr.set_line_width (1)
            cr.move_to(vec.p.x,vec.p.y)
            cr.line_to(vec.p2.x,vec.p2.y)
            cr.stroke()

        if src_pt is not None:
            cr.set_source_rgb (1,0,0)
            cr.move_to(src_pt.x, src_pt.y)
            cr.arc(src_pt.x, src_pt.y, 2, 0, 2.0 * math.pi)
            cr.fill()

        if trg_pt is not None:
            cr.set_source_rgb (0,1,0)
            cr.move_to(trg_pt.x, trg_pt.y)
            cr.arc(trg_pt.x, trg_pt.y, 2, 0, 2.0 * math.pi)
            cr.fill()

    def _on_configure_event(self, widget, event):
        allocation = self.get_allocation()
        self._surface = self.get_window().create_similar_surface(
                                            cairo.CONTENT_COLOR,
                                            allocation.width,
                                            allocation.height)

        self._draw_background()

    def get_vec_and_points(self):
        vec = None
        src_pt = None
        trg_pt = self._model.moving_pt if self._model else None
        if self._model and self._mousex is not None:
            src_pt = euclid.Point2(self._mousex,self._mousey)
            if self._snap_to_moving_pt and self._model.moving_pt is not None:
                vec = self._model.connect_to_moving_point(p=None,
                                    px=self._mousex,py=self._mousey)
            else:
                vec = self._model.connect_closest(p=None,
                                    px=self._mousex,py=self._mousey)
        return vec,src_pt,trg_pt

    def move_along(self, scale, wrap=False):
        val = self._model.move_point(scale, wrap=wrap)
        self.queue_draw()
        return val

