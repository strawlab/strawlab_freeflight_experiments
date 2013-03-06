#!/usr/bin/env python

import math
import threading

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')
import rospy
from std_msgs.msg import String, UInt32, Bool
from geometry_msgs.msg import Vector3

from flyflypath.transform import SVGTransform
from flyflypath.euclid import Point2, LineSegment2

import cairo
from gi.repository import Gtk, Gdk, GLib, GObject

GObject.threads_init()
Gdk.threads_init()

class FlycaveWidget(Gtk.DrawingArea):
    def __init__(self,w_px,r_m):
        Gtk.DrawingArea.__init__(self)

        self._xform = SVGTransform(w=w_px)
        self._r_px = r_m * w_px

        self.set_size_request(w_px,h_px)
        self.add_events(
                Gdk.EventMask.POINTER_MOTION_HINT_MASK | \
                Gdk.EventMask.POINTER_MOTION_MASK)
        self.connect('motion-notify-event', self._on_motion_notify_event)
        self.connect('draw', self._on_draw_event)
        self.connect('configure-event', self._on_configure_event)

        self._surface = None
        self._mousex = self._mousey = None

    def _draw_background(self):
        cr = cairo.Context(self._surface)
        cr.set_source_rgb(1, 1, 1)
        cr.paint()

        cr.set_source_rgb (0, 0, 0)
        cr.set_line_width (1)
        cx,cy = self._xform.xy_to_pxpy(0,0)
        cr.arc(cx, cy, self._r_px, 0, 2.0 * math.pi)
        cr.stroke()
        
    def _on_motion_notify_event(self, da, event):
        (window, self._mousex, self._mousey, state) = event.window.get_pointer()
        self.queue_draw()
        return True

    def _on_draw_event(self, widget, cr):
        cr.set_source_surface(self._surface, 0.0, 0.0)
        cr.paint()

        vec,pts = self.get_vec_and_points()
        
        if vec is not None:
            cr.set_source_rgb (1, 0, 0)
            cr.set_line_width (1)
            cr.move_to(vec.p.x,vec.p.y)
            cr.line_to(vec.p2.x,vec.p2.y)
            cr.stroke()

        for pt,rgb in pts:
            if pt is not None:
                cr.set_source_rgb (*rgb)
                cr.move_to(pt.x, pt.y)
                cr.arc(pt.x, pt.y, 2, 0, 2.0 * math.pi)
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
        FlycaveWidget.__init__(self, 500, 0.5)

        self._src = None
        self._trg = None
        self._trg_prev = None
        self._vec = None
        self._post = None

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
        rospy.Subscriber("fixation/post",
                         Vector3,
                         self._on_post_move)

        self._w.show_all()

        GLib.timeout_add(1000/20, self._redraw)

    def _quit(self, *args):
        rospy.signal_shutdown("quit")
        Gtk.main_quit()

    def _redraw(self):
        self.queue_draw()
        return True

    def _on_source_move(self, msg):
        x,y = self._xform.xy_to_pxpy(msg.x,msg.y)
        self._src = Point2(int(x),int(y))

    def _on_target_move(self, msg):
        x,y = self._xform.xy_to_pxpy(msg.x,msg.y)
        self._trg = Point2(int(x),int(y))

    def _on_post_move(self, msg):
        x,y = self._xform.xy_to_pxpy(msg.x,msg.y)
        self._post = Point2(int(x),int(y))

    def get_vec_and_points(self):
        return self._vec,(
                (self._src,(1,0,0)),
                (self._trg,(0,1,0)),
                (self._post,(0,0,1)))

if __name__ == "__main__":
    rospy.init_node("fixation_monitor")
    ui = FixationWidget()
    Gtk.main()
