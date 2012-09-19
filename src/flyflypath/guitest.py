import sys
import xml.dom.minidom

import svg
import polyline
import euclid

import cairo
from gi.repository import Gtk, Gdk

class PixelCoordWidget(Gtk.DrawingArea):
    def __init__(self, path):
        Gtk.DrawingArea.__init__(self)
        self.set_size_request(500,500)
        self.add_events(
                Gdk.EventMask.POINTER_MOTION_HINT_MASK | \
                Gdk.EventMask.POINTER_MOTION_MASK)
        self.connect('motion-notify-event', self.on_motion_notify_event)
        
        self._surface = None
        self._model = None

        self._points_to_path = None
        
        #parse the SVG
        d = xml.dom.minidom.parse(open(path,'r'))
        paths = d.getElementsByTagName('path')
        assert len(paths) == 1
        pathdata = str(paths[0].getAttribute('d'))
        self._svgiter = svg.PathIterator(pathdata)

        #we can build the line model now, but we need to wait until the widget
        #is shown befor building the background
        self._build_linemodel()
        
    def _build_linemodel(self):
        self._model = polyline.polyline_from_svg_path(self._svgiter)

    def _draw_background(self):
        cr = cairo.Context(self._surface)
        cr.set_source_rgb(1, 1, 1)
        cr.paint()

        cr.set_source_rgb (0, 0, 0)
        cr.set_line_width (1)
        for path_element in self._svgiter:
            svg.draw_on_context(cr, path_element)
        cr.stroke()

    def on_motion_notify_event(self, da, event):
        (window, x, y, state) = event.window.get_pointer()
        closest = self._model.connect( euclid.Point2(x,y) )

        self._points_to_path = (x,y,closest.point.x,closest.point.y)
        self.queue_draw()

        return True

    def do_draw(self, cr):
        cr.set_source_surface(self._surface, 0.0, 0.0)
        cr.paint()
        
        if self._points_to_path:
            x1,y1,x2,y2 = self._points_to_path
            cr.set_source_rgb (1, 0, 0)
            cr.set_line_width (1)
            cr.move_to(x1,y1)
            cr.line_to(x2,y2)
            cr.stroke()

    def do_configure_event(self, event):
        allocation = self.get_allocation()
        self._surface = self.get_window().create_similar_surface(
                                            cairo.CONTENT_COLOR,
                                            allocation.width,
                                            allocation.height)

        self._draw_background()

class Tester:
    def __init__(self, path):
        self._w = Gtk.Window()
        self._w.connect("delete-event", Gtk.main_quit)
        self._view = PixelCoordWidget(path)
        self._w.add(self._view)
        self._w.show_all()

if __name__ == "__main__":
    t = Tester(sys.argv[1] if len(sys.argv) > 1 else "plain.svg")
    Gtk.main()
