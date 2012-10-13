#!/usr/bin/env python

import math
import threading

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')
import rospy
from std_msgs.msg import String, UInt32, Bool
from geometry_msgs.msg import Vector3

from flyflypath.euclid import Point2, LineSegment2

import cairo
from gi.repository import Gtk, Gdk, GLib, GObject

GObject.threads_init()
Gdk.threads_init()

class FlycaveWidget(Gtk.DrawingArea):
    def __init__(self,w_px,h_px,r_m):
        Gtk.DrawingArea.__init__(self)

        assert w_px == h_px

        self._w_px = w_px
        self._h_px = h_px
        self._r_px = r_m * self._w_px

        self.set_size_request(self._w_px,self._h_px)
        self.add_events(
                Gdk.EventMask.POINTER_MOTION_HINT_MASK | \
                Gdk.EventMask.POINTER_MOTION_MASK)
        self.connect('motion-notify-event', self._on_motion_notify_event)
        self.connect('draw', self._on_draw_event)
        self.connect('configure-event', self._on_configure_event)

        self._surface = None
        self._mousex = self._mousey = None

    def _xy_to_pxpy(self, x, y):
        x = (x * self._w_px) + (self._w_px/2.0)
        y = (y * self._h_px) + (self._h_px/2.0)
        return x,y
       
    def _draw_background(self):
        cr = cairo.Context(self._surface)
        cr.set_source_rgb(1, 1, 1)
        cr.paint()

        cr.set_source_rgb (0, 0, 0)
        cr.set_line_width (1)
        cx,cy = self._xy_to_pxpy(0,0)
        cr.arc(cx, cy, self._r_px, 0, 2.0 * math.pi)
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

class FixationWidget(FlycaveWidget):
    def __init__(self):
        FlycaveWidget.__init__(self, 500, 500, 0.5)

        self._src = None
        self._trg = None
        self._vec = None

        t = threading.Thread(target=rospy.spin).start()
        self._lock = threading.Lock()

        self._w = Gtk.Window()
        self._w.connect("delete-event", self._quit)
        self._w.add(self)

        rospy.Subscriber("fixation/fly",
                         Vector3,
                         self._on_source_move)
        rospy.Subscriber("fixation/target",
                         Vector3,
                         self._on_target_move)

        self._w.show_all()

        GLib.timeout_add(1000/20, self._redraw)

    def _quit(self, *args):
        rospy.signal_shutdown("quit")
        Gtk.main_quit()

    def _redraw(self):
        self.queue_draw()
        return True

    def _on_source_move(self, msg):
        x,y = self._xy_to_pxpy(msg.x,msg.y)
        self._src = Point2(int(x),int(y))

    def _on_target_move(self, msg):
        x,y = self._xy_to_pxpy(msg.x,msg.y)
        self._trg = Point2(int(x),int(y))

    def get_vec_and_points(self):
        return self._vec,self._src, self._trg

if __name__ == "__main__":
    rospy.init_node("fixation_monitor")
    ui = FixationWidget()
    Gtk.main()
