#!/usr/bin/env python

import math
import numpy as np
import threading
import argparse
import os.path
import collections

PACKAGE='strawlab_freeflight_experiments'

import roslib
import roslib.packages
roslib.load_manifest(PACKAGE)

import rospy
import flyvr.display_client as display_client
from std_msgs.msg import UInt32, Bool, Float32, String
from geometry_msgs.msg import Vector3, Pose, Polygon
from ros_flydra.msg import flydra_mainbrain_super_packet

import flyflypath.transform
import flyflypath.model
import nodelib.node
import nodelib.visualization
import strawlab_freeflight_experiments.conditions as sfe_conditions

from ros_flydra.constants import IMPOSSIBLE_OBJ_ID

Pos = collections.namedtuple('Pos', ('x', 'y', 'z'))

pkg_dir = roslib.packages.get_pkg_dir(PACKAGE)

CONTROL_RATE        = 40.0      #Hz

HOLD_COND = "midgray.osg"

FLY_DIST_CHECK_TIME = 5.0
FLY_DIST_MIN_DIST   = 0.2

START_ZDIST     = 0.4
START_Z         = 0.5

# z range for fly tracking (dropped outside)
Z_MINIMUM = 0.00
Z_MAXIMUM = 0.95

WRAP_MODEL_H_Z      = 5.0

TIMEOUT             = 0.5

XFORM = flyflypath.transform.SVGTransform()

class Node(nodelib.node.Experiment):
    def __init__(self, args):
        super(Node, self).__init__(args=args,
                                   state=("stimulus_filename","svg_filename","startr","stopr","startbuf","stopbuf","x0","y0","z0","sx","sy","sz"))

        self._pub_stim_mode = display_client.DisplayServerProxy.set_stimulus_mode(
            'StimulusOSGFile')

        self.pub_stimulus = rospy.Publisher('stimulus_filename', String, latch=True, tcp_nodelay=True)
        self.pub_lag = rospy.Publisher('extra_lag_msec', Float32, latch=True, tcp_nodelay=True)
        self.pub_model_pose = rospy.Publisher('model_pose', Pose, latch=True, tcp_nodelay=True)
        self.pub_model_scale = rospy.Publisher('model_scale', Vector3, latch=True, tcp_nodelay=True)

        self.trigarea_pub = rospy.Publisher('trigger_area', Polygon, latch=True, tcp_nodelay=True)

        self.pub_lock_object = rospy.Publisher('lock_object', UInt32, latch=True, tcp_nodelay=True)
        self.pub_lock_object.publish(IMPOSSIBLE_OBJ_ID)

        #protect the tracked id and fly position between the time syncronous main loop and the asyn
        #tracking/lockon/off updates
        self.trackinglock = threading.Lock()
        with self.trackinglock:
            self.currently_locked_obj_id = None
            self.fly = Vector3()
            self.flyv = Vector3()
            self.framenumber = 0
            now = rospy.get_time()
            self.first_seen_time = now
            self.last_seen_time = now
            self.last_fly_x = self.fly.x; self.last_fly_y = self.fly.y; self.last_fly_z = self.fly.z;
            self.last_check_flying_time = now
            self.fly_dist = 0

        self.svg_pub = rospy.Publisher("svg_filename", String, latch=True)
        self.src_pub = rospy.Publisher("source", Vector3)
        self.ack_pub = rospy.Publisher("active", Bool)

        self._z00 = 0.0
        self.switch_conditions()

        rospy.Subscriber("flydra_mainbrain/super_packets",
                         flydra_mainbrain_super_packet,
                         self.on_flydra_mainbrain_super_packets)

    def get_model_pose_msg(self):
        msg = Pose()
        msg.position.x = self.x0
        msg.position.y = self.y0
        msg.position.z = self.z0
        msg.orientation.w = 1
        return msg

    def _get_trigger_area(self):
        if self.hitm_start is not None:
            x,y = self.hitm_start.points
            return nodelib.visualization.get_trigger_volume_polygon(XFORM,zip(x,y))
        else:
            return nodelib.visualization.get_circle_trigger_volume_polygon(
                                        XFORM,
                                        self.startr,self.x0,self.y0)

    def _get_hitmanager(self, path, scale):
        #try the faster single path case first
        try:
            mod = flyflypath.model.SvgPath(path)
        except flyflypath.model.MultiplePathSvgError:
            mod = flyflypath.model.MultipleSvgPath(path)
        return mod.get_hitmanager(XFORM, validate=True, scale=scale)

    def switch_conditions(self):

        self.drop_lock_on()

        self.stimulus_filename = str(self.condition['stimulus_filename'])
        self.x0                = float(self.condition['x0'])
        self.y0                = float(self.condition['y0'])
        try:
            self._z00          = float(self.condition['z0'])
        except KeyError:
            self._z00          = 0.0

        self.lag               = float(self.condition.get('lag',0.0))

        sx =                float(self.condition.get('sx',1.0))
        sy =                float(self.condition.get('sy',1.0))
        sz =                float(self.condition.get('sz',1.0))
        self.pub_model_scale.publish(sx,sy,sz)

        try:
            self.startr        = float(self.condition['start_radius'])
        except KeyError:
            self.startr        = None
        try:
            self.stopr         = float(self.condition['stop_radius'])
        except KeyError:
            self.stopr         = None

        try:
            #we can optionally move the model down vertically
            self.model_pose_voffset = float(self.condition['model_pose_voffset'])
            if self.model_pose_voffset > 0:
                raise Exception("NOT TESTED")
        except:
            self.model_pose_voffset = 0.0

        #settings related to the svg path which defines the confinement or lock on/off region
        try:
            svg_filename       = str(self.condition['svg_filename'])
            svg_path = os.path.join(pkg_dir,"data","svgpaths",svg_filename)
        except KeyError:
            #conditional and backwards compatible handling for specifying start and
            #stop conditions relative to a buffer around the svg
            svg_path = os.path.join(pkg_dir,"data","svgpaths",self.stimulus_filename[:-4])
            if os.path.isfile(svg_path):
                svg_filename = os.path.basename(svg_path)
            else:
                svg_filename = None

        #settings related to a lock on region defined as a buffer around svg_filename
        try:
            startbuf            = float(self.condition['start_buffer'])
            self.hitm_start     = self._get_hitmanager(svg_path, startbuf)
        except (KeyError,ValueError,flyflypath.model.SvgError), e:
            startbuf            = None
            self.hitm_start     = None

        #settings related to a lock off region defined as a buffer around svg_filename
        try:
            stopbuf             = float(self.condition['stop_buffer'])
            self.hitm_stop      = self._get_hitmanager(svg_path, stopbuf)
        except (KeyError,ValueError,flyflypath.model.SvgError):
            stopbuf             = None
            self.hitm_stop      = None

        #settings related to a hiding the stimulus when the fly exceeds a
        #region defined as a buffer around svg_filename
        try:
            hidebuf             = float(self.condition['hide_buffer'])
            self.hitm_hide      = self._get_hitmanager(svg_path, hidebuf)
        except (KeyError,ValueError,flyflypath.model.SvgError):
            hidebuf             = None
            self.hitm_hide      = None

        desc = "start: %s stop: %s" % (('r=%s' % self.startr) if self.startr is not None else \
                                       ('%s +/- %s' % (svg_filename, startbuf if startbuf is not None else 0)),
                                       ('r=%s' % self.stopr) if self.stopr is not None else \
                                       ('%s +/- %s' % (svg_filename, stopbuf if stopbuf is not None else 0)))

        self.log.x0 = self.x0
        self.log.y0 = self.y0

        self.log.stimulus_filename = self.stimulus_filename
        self.log.svg_filename = svg_filename
        self.log.startr = self.startr
        self.log.stopr = self.stopr
        self.log.startbuf = startbuf
        self.log.stopbuf = stopbuf

        self.pub_lag.publish(self.lag)
        self.pub_model_pose.publish( self.get_model_pose_msg() )
        self.svg_pub.publish(svg_path)

        self.trigarea_pub.publish( self._get_trigger_area() )

        rospy.loginfo('condition: %s (%f,%f) %s' % (self.condition.name,self.x0,self.y0,desc))


    def run(self):
        rospy.loginfo('running stimulus')
        r = rospy.Rate(CONTROL_RATE)
        while not rospy.is_shutdown():
            with self.trackinglock:
                currently_locked_obj_id = self.currently_locked_obj_id
                fly_x = self.fly.x; fly_y = self.fly.y; fly_z = self.fly.z
                fly_vx = self.flyv.x; fly_vy = self.flyv.y; fly_vz = self.flyv.z
                framenumber = self.framenumber

            if currently_locked_obj_id is None:
                active = False
            else:
                now = rospy.get_time()
                if now-self.last_seen_time > TIMEOUT:
                    self.drop_lock_on('timeout')
                    rospy.loginfo('TIMEOUT: time since last seen >%.1fs' % (TIMEOUT))
                    continue

                if (fly_z > Z_MAXIMUM) or (fly_z < Z_MINIMUM):
                    self.drop_lock_on('z range')
                    continue

                if np.isnan(fly_x):
                    #we have a race  - a fly to track with no pose yet
                    continue

                if self.has_left_stop_volume(Pos(fly_x,fly_y,fly_z)):
                    self.drop_lock_on('left volume')
                    continue

                active = True

                #distance accounting, give up on fly if it is not moving
                self.fly_dist += math.sqrt((fly_x-self.last_fly_x)**2 +
                                           (fly_y-self.last_fly_y)**2 +
                                           (fly_z-self.last_fly_z)**2)
                self.last_fly_x = fly_x; self.last_fly_y = fly_y; self.last_fly_z = fly_z;

                # drop slow moving flies
                if now-self.last_check_flying_time > FLY_DIST_CHECK_TIME:
                    fly_dist = self.fly_dist
                    self.last_check_flying_time = now
                    self.fly_dist = 0
                    if fly_dist < FLY_DIST_MIN_DIST: # drop fly if it does not move enough
                        self.drop_lock_on('slow')
                        continue

                px,py = XFORM.xy_to_pxpy(fly_x,fly_y)
                self.src_pub.publish(px,py,fly_z)

            if active:

                if self.hitm_hide is not None:
                    if not self.hitm_hide.contains(fly_x, fly_y):
                        self.pub_stimulus.publish( HOLD_COND )

                dz = self.model_pose_voffset / CONTROL_RATE
                self.z0 = (self.z0 + dz) % -WRAP_MODEL_H_Z
                self.pub_model_pose.publish( self.get_model_pose_msg() )

                self.log.z0 = self.z0

            #new combine needs data recorded at the framerate
            self.log.framenumber = framenumber
            self.log.update()

            self.ack_pub.publish(active)

            r.sleep()

        rospy.loginfo('%s finished. saved data to %s' % (rospy.get_name(), self.log.close()))

    def _is_in_circle_at_origin(self,pos,rad):
        c = np.array( (self.x0,self.y0) )
        p = np.array( (pos.x, pos.y) )
        dist = np.sqrt(np.sum((c-p)**2))
        if (dist < rad) and (abs(pos.z-START_Z) < START_ZDIST):
            return True
        return False

    def is_in_trigger_volume(self,pos):
        if self.hitm_start is not None:
            return self.hitm_start.contains(pos.x, pos.y)
        else:
            return self._is_in_circle_at_origin(pos,self.startr)

    def has_left_stop_volume(self,pos):
        if self.hitm_stop is not None:
            return not self.hitm_stop.contains(pos.x, pos.y)
        else:
            if self.stopr is not None:
                return not self._is_in_circle_at_origin(pos,self.stopr)
            else:
                return False

    def on_flydra_mainbrain_super_packets(self,data):
        now = rospy.get_time()
        for packet in data.packets:
            for obj in packet.objects:
                if self.currently_locked_obj_id is not None:
                    if obj.obj_id == self.currently_locked_obj_id:
                        self.last_seen_time = now
                        self.fly = obj.position
                        self.framenumber = packet.framenumber
                else:
                    if self.is_in_trigger_volume(obj.position):
                        self.lock_on(obj,packet.framenumber)

    def update(self):
        self.log.update()
        self.pub_lock_object.publish( self.log.lock_object )

    def lock_on(self,obj,framenumber):
        with self.trackinglock:
            rospy.loginfo('locked object %d at frame %d' % (obj.obj_id,framenumber))
            now = rospy.get_time()
            self.currently_locked_obj_id = obj.obj_id
            self.last_seen_time = now
            self.first_seen_time = now
            self.log.lock_object = obj.obj_id
            self.log.framenumber = framenumber
            self.last_check_flying_time = now
            self.fly = obj.position

        self.z0 = self._z00
        self.log.z0 = self.z0

        self.pub_model_pose.publish( self.get_model_pose_msg() )        
        self.pub_stimulus.publish( self.stimulus_filename )
        self.update()

    def drop_lock_on(self, reason=''):
        with self.trackinglock:
            old_id = self.currently_locked_obj_id
            now = rospy.get_time()
            dt = now - self.first_seen_time

            rospy.loginfo('dropping locked object %s %s (tracked for %.1f)' % (old_id,reason,dt))

            self.currently_locked_obj_id = None

            self.log.lock_object = IMPOSSIBLE_OBJ_ID
            self.log.framenumber = 0

        if (dt > 30) and (old_id is not None):
            self.save_cool_condition(old_id, note="Fly %s confined for %.1fs" % (old_id, dt))

        self.z0 = self._z00
        self.log.z0 = self.z0

        self.pub_stimulus.publish( HOLD_COND )
        self.update()        

def main():
    rospy.init_node("confinement")
    parser, args = nodelib.node.get_and_parse_commandline()
    node = Node(args)
    return node.run()

if __name__=='__main__':
    main()
