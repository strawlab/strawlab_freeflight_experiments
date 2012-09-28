import sys
import xml.dom.minidom

import svg
import polyline
import euclid
import math
import path

import cairo
from gi.repository import Gtk, Gdk, GLib

class PixelCoordWidget(Gtk.DrawingArea):
    def __init__(self, svgpath, to_pt=False):
        Gtk.DrawingArea.__init__(self)
        self.set_size_request(500,500)
        self.add_events(
                Gdk.EventMask.POINTER_MOTION_HINT_MASK | \
                Gdk.EventMask.POINTER_MOTION_MASK)
        self.connect('motion-notify-event', self.on_motion_notify_event)

        self._snap_to_moving_pt = to_pt

        self._surface = None
        self._polyline = None

        self._moving_pt = None
        self._points_to_path = None
        self._mousex = self._mousey = None
       
        self._model = path.SvgPath(svgpath)
        self._polyline = self._model.polyline
        self._svgiter = self._model.svgiter
        
    def _draw_background(self):
        cr = cairo.Context(self._surface)
        cr.set_source_rgb(1, 1, 1)
        cr.paint()

        cr.set_source_rgb (0, 0, 0)
        cr.set_line_width (1)
        for path_element in self._svgiter:
            svg.draw_on_context(cr, path_element)
        cr.stroke()
        
        #draw the approximation
        cr.set_source_rgb (0, 0, 1)
        cr.set_line_width (0.3)
        cr.move_to(self._polyline.points[0].x,self._polyline.points[0].y)
        for i in range(1,len(self._polyline.points)):
            cr.line_to(self._polyline.points[i].x,self._polyline.points[i].y)
        cr.stroke()

    def on_motion_notify_event(self, da, event):
        (window, self._mousex, self._mousey, state) = event.window.get_pointer()
        self.queue_draw()
        return True

    def do_draw(self, cr):
        cr.set_source_surface(self._surface, 0.0, 0.0)
        cr.paint()

        if self._mousex is not None:
            if self._snap_to_moving_pt and self._moving_pt is not None:
                self._points_to_path = (self._mousex,self._mousey,self._moving_pt.x,self._moving_pt.y)
            else:
                closest = self._polyline.connect( euclid.Point2(self._mousex,self._mousey) )
                self._points_to_path = (self._mousex,self._mousey,closest.point.x,closest.point.y)
        
        if self._points_to_path:
            x1,y1,x2,y2 = self._points_to_path
            cr.set_source_rgb (1, 0, 0)
            cr.set_line_width (1)
            cr.move_to(x1,y1)
            cr.line_to(x2,y2)
            cr.stroke()

        if self._moving_pt is not None:
            cr.set_source_rgb (0,1,0)
            cr.move_to(self._moving_pt.x, self._moving_pt.y)
            cr.arc(self._moving_pt.x, self._moving_pt.y, 2, 0, 2.0 * math.pi)
            cr.fill()

    def do_configure_event(self, event):
        allocation = self.get_allocation()
        self._surface = self.get_window().create_similar_surface(
                                            cairo.CONTENT_COLOR,
                                            allocation.width,
                                            allocation.height)

        self._draw_background()

    def move_along(self, scale):
        scale = min(1.0,max(0.0,scale))
        self._moving_pt = self._polyline.along(scale)
        self.queue_draw()

class Tester:
    def __init__(self, path):
        self._w = Gtk.Window()
        self._w.connect("delete-event", Gtk.main_quit)
        self._view = PixelCoordWidget(path, to_pt=True)
        self._w.add(self._view)
        self._w.show_all()

        self._step = 0.01
        self._along = 0.0
        GLib.timeout_add(500, self._move_along)

    def _move_along(self):
        self._along += self._step
        self._view.move_along(self._along)
        return self._along <= 1.0



if __name__ == "__main__":
    t = Tester(sys.argv[1] if len(sys.argv) > 1 else "plain.svg")
    Gtk.main()
