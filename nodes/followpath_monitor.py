#!/usr/bin/env python

import threading

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')
import rospy
from std_msgs.msg import String, UInt32, Bool
from geometry_msgs.msg import Vector3, Polygon, Point, Point32

import flyflypath.model
import flyflypath.view
import flyflypath.transform
from flyflypath.euclid import Point2, LineSegment2

from gi.repository import Gtk, Gdk, GLib, GObject

GObject.threads_init()
Gdk.threads_init()

XFORM = flyflypath.transform.SVGTransform()

class RemoveSvgWidget(flyflypath.view.SvgPathWidget):
    def __init__(self):
        flyflypath.view.SvgPathWidget.__init__(self, None)

        self._src = None
        self._trg = None
        self._active = False
        self._area = []
        self._path = []

        t = threading.Thread(target=rospy.spin).start()
        self._lock = threading.Lock()

        self._w = Gtk.Window()
        self._w.connect("delete-event", self._quit)
        self._w.add(self)

        rospy.Subscriber("active",
                         Bool,
                         self._on_active)

        #all in pixels
        rospy.Subscriber("svg_filename",
                         String,
                         self._on_svg_filename)
        rospy.Subscriber("source",
                         Vector3,
                         self._on_source_move)
        rospy.Subscriber("target",
                         Vector3,
                         self._on_target_move)
        rospy.Subscriber("trigger_area",
                         Polygon,
                         self._on_trigger_area)
        rospy.Subscriber("path",
                         Polygon,
                         self._on_path)

        #all in meters
        rospy.Subscriber("source_m",
                         Vector3,
                         lambda msg: self._on_source_move(self._to_m_vec3(msg)))
        rospy.Subscriber("target_m",
                         Vector3,
                         lambda msg: self._on_target_move(self._to_m_vec3(msg)))
        rospy.Subscriber("trigger_area_m",
                         Polygon,
                         lambda msg: self._on_trigger_area(self._to_m_polygon(msg)))
        rospy.Subscriber("path_m",
                         Polygon,
                         lambda msg: self._on_path(self._to_m_polygon(msg)))


        self._w.show_all()

        GLib.timeout_add(1000/20, self.redraw)

    def _to_m_vec3(self, msg):
        return Vector3(*XFORM.xyz_to_pxpypz(msg.x,msg.y,msg.z))

    def _to_m_polygon(self, msg):
        return Polygon(points=[Point32(*XFORM.xyz_to_pxpypz(p.x,p.y,p.z)) for p in msg.points])

    def _quit(self, *args):
        rospy.signal_shutdown("quit")
        Gtk.main_quit()

    def _on_path(self, msg):
        self._path = tuple(Point2(p.x,p.y) for p in msg.points)

    def _on_trigger_area(self, msg):
        self._area = tuple(Point2(p.x,p.y) for p in msg.points)

    def _on_svg_filename(self, msg):
        with self._lock:
            try:
                self._model = flyflypath.model.MovingPointSvgPath(msg.data)
            except flyflypath.model.SvgError, e:
                self._model = None
            self.draw_background()

    def _on_source_move(self, msg):
        try:
            self._src = Point(int(msg.x),int(msg.y),msg.z)
        except:
            pass

    def _on_target_move(self, msg):
        try:
            self._trg = Point(int(msg.x),int(msg.y),msg.z)
        except:
            pass

    def _on_active(self, msg):
        self._active = msg.data

    def get_vec_and_points(self):
        vecs = []
        with self._lock:
            src_pt = self._src
            trg_pt = self._trg
        if self._src is not None and self._trg is not None:
            try:
                vecs.append( (LineSegment2(
                                Point2(self._src.x,self._src.y),
                                Point2(self._trg.x,self._trg.y)),
                             (1,0,0))
                )
            except AttributeError:
                #zero length line
                pass

        pts = [ (trg_pt,(0,1,0)) ]
        if self._active:
            pts.append( (src_pt,(1,0,0)) )

        return vecs,pts,[(self._area,(0,0.35,0)), (self._path,(0,0,0))]

if __name__ == "__main__":
    rospy.init_node("followpath_monitor")
    ui = RemoveSvgWidget()
    Gtk.main()
