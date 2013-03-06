#!/usr/bin/env python

import threading

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')
import rospy
from std_msgs.msg import String, UInt32, Bool
from geometry_msgs.msg import Vector3, PoseArray

import flyflypath.model
import flyflypath.view
from flyflypath.euclid import Point2, LineSegment2

from gi.repository import Gtk, Gdk, GLib, GObject

GObject.threads_init()
Gdk.threads_init()

class RemoveSvgWidget(flyflypath.view.SvgPathWidget):
    def __init__(self):
        flyflypath.view.SvgPathWidget.__init__(self, None)

        self._src = None
        self._trg = None
        self._active = False
        self._area = []

        t = threading.Thread(target=rospy.spin).start()
        self._lock = threading.Lock()

        self._w = Gtk.Window()
        self._w.connect("delete-event", self._quit)
        self._w.add(self)

        rospy.Subscriber("svg_filename",
                         String,
                         self._on_svg_filename)
        rospy.Subscriber("source",
                         Vector3,
                         self._on_source_move)
        rospy.Subscriber("target",
                         Vector3,
                         self._on_target_move)
        rospy.Subscriber("active",
                         Bool,
                         self._on_active)
        rospy.Subscriber("trigger_area",
                         PoseArray,
                         self._on_trigger_area)

        self._w.show_all()

        GLib.timeout_add(1000/20, self._redraw)

    def _quit(self, *args):
        rospy.signal_shutdown("quit")
        Gtk.main_quit()

    def _redraw(self):
        self.queue_draw()
        return True

    def _on_trigger_area(self, msg):
        self._area = [(p.position.x,p.position.y) for p in msg.poses]

    def _on_svg_filename(self, msg):
        with self._lock:
            try:
                self._model = flyflypath.model.MovingPointSvgPath(msg.data)
            except flyflypath.model.SvgError:
                self._model = None
            self._draw_background()

    def _on_source_move(self, msg):
        try:
            self._src = Point2(int(msg.x),int(msg.y))
        except:
            pass

    def _on_target_move(self, msg):
        try:
            self._trg = Point2(int(msg.x),int(msg.y))
        except:
            pass

    def _on_active(self, msg):
        self._active = msg.data

    def get_vec_and_points(self):
        vec = None
        src_pt = self._src
        trg_pt = self._trg
        if self._src is not None and self._trg is not None:
            with self._lock:
                vec = LineSegment2(self._src,self._trg)

        pts = [ (trg_pt,(0,1,0)) ]
        if self._active:
            pts.append( (src_pt,(1,0,0)) )

        return vec,pts,tuple(self._area)

if __name__ == "__main__":
    rospy.init_node("followpath_monitor")
    ui = RemoveSvgWidget()
    Gtk.main()
