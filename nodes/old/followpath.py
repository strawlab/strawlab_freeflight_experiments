#!/usr/bin/env python

import os
import numpy as np
import time
import threading
import argparse
import math

PACKAGE='strawlab_freeflight_experiments'

import roslib
import roslib.packages
roslib.load_manifest(PACKAGE)

import rospy
import flyvr.display_client as display_client
from std_msgs.msg import String, UInt32, Bool
from geometry_msgs.msg import Vector3, Pose
from ros_flydra.msg import flydra_mainbrain_super_packet

import flyflypath.transform
import flyflypath.model
import flyflypath.view
import flyflypath.polyline
import flyflypath.euclid
import nodelib.log
from ros_flydra.constants import IMPOSSIBLE_OBJ_ID, IMPOSSIBLE_OBJ_ID_ZERO_POSE

from strawlab_freeflight_experiments.topics import *

pkg_dir = roslib.packages.get_pkg_dir(PACKAGE)

PATH_TO_FOLLOW      = os.path.join(pkg_dir,"data","svgpaths","infinity.svg")
CONTROL_RATE        = 40.0      #Hz
MOVING_POINT_TIME   = 25.0      #15s for the target to move along the path (fly move at 0.1m/s)

SWITCH_MODE_TIME    = 5.0*60    #alternate between control and static (i.e. experimental control) seconds

Z_TARGET = 0.5

PX_DIST_ADVANCE = 30

SVG_WRAP_PATH = True

FLY_DIST_CHECK_TIME = 5.0
FLY_DIST_MIN_DIST = 0.2

TIMEOUT = 0.5

#FIXME: shrink made non-zero to keep away from edge
XFORM = flyflypath.transform.SVGTransform(shrink=0.8)

CONDITIONS = ("nolock/+0.000",
              "follow+control/+0.060",
              "follow+control/+0.000",
              "follow+control/-0.060")
START_CONDITION = CONDITIONS[1]

def is_stepwise_mode(condition):
    return condition.startswith("follow+stepwise")

def is_control_mode(condition):
    return condition.startswith("follow+control")

def is_control_nolock_mode(condition):
    return condition.startswith("nolock")

class Logger(nodelib.log.CsvLogger):
    STATE = ("svg_filename","condition","src_x","src_y","src_z","target_x","target_y","target_z",
                 "stim_x","stim_y","stim_z","move_ratio","active","lock_object","framenumber")

class Node(object):
    def __init__(self, wait_for_flydra, use_tmpdir):

        #for x,y in ((0.2,0.3),(-0.2,0.4),(0.3,-0.1),(-0.3,-0.45),(0.5,0.5),(0,0)):
        #    px,py = XFORM.xy_to_pxpy(x,y)

        self._pub_stim_mode = display_client.DisplayServerProxy.set_stimulus_mode(
            'StimulusCUDAStarFieldAndModel')

        self.starfield_velocity_pub = rospy.Publisher(TOPIC_STAR_VELOCITY, Vector3, latch=True, tcp_nodelay=True)
        self.starfield_velocity_pub.publish(Vector3())
        self.starfield_post_pub = rospy.Publisher(TOPIC_MODEL_POSITION, Pose, latch=True, tcp_nodelay=True)
        self.starfield_post_pub.publish(self.get_hide_post_msg())
        self.lock_object = rospy.Publisher('lock_object', UInt32, latch=True, tcp_nodelay=True)
        self.lock_object.publish(IMPOSSIBLE_OBJ_ID_ZERO_POSE)

        self.model = flyflypath.model.MovingPointSvgPath(PATH_TO_FOLLOW)

        self.log = Logger(wait=wait_for_flydra, use_tmpdir=use_tmpdir)

        startpt = self.model.polyline.p
        self.start_x, self.start_y = XFORM.pxpy_to_xy(startpt.x,startpt.y)

        self.p_const = 0.0

        #protect the traked id and fly position between the time syncronous main loop and the asyn
        #tracking/lockon/off updates
        self.trackinglock = threading.Lock()
        self.currently_locked_obj_id = None
        self.fly = Vector3()
        self.framenumber = 0
        now = rospy.get_time()
        self.first_seen_time = now
        self.last_seen_time = now
        self.last_fly_x = self.fly.x; self.last_fly_y = self.fly.y; self.last_fly_z = self.fly.z;
        self.last_check_flying_time = now
        self.fly_dist = 0

        self.moving_ratio = 0.0
        self.switch_conditions(None,force=START_CONDITION)

        #init the log to zero (we publish immediately)
        self.log.svg_filename = self.model.svgpath
        self.log.src_x = self.fly.x; self.log.src_y = self.fly.y; self.log.src_z = self.fly.z;
        target = flyflypath.polyline.ZeroLineSegment2()
        self.log.target_x = target.p2.x; self.log.target_y = target.p2.y; self.log.target_z = 0.0
        vec = Vector3()
        self.log.stim_x = vec.x; self.log.stim_y = vec.y; self.log.stim_z = 0.0
        self.log.move_ratio = self.moving_ratio
        self.log.lock_object = IMPOSSIBLE_OBJ_ID_ZERO_POSE
        self.log.framenumber = self.framenumber

        #publish for the follow_path monitor
        self.svg_pub = rospy.Publisher('svg_filename', String, latch=True, tcp_nodelay=True)
        self.srcpx_pub = rospy.Publisher('source', Vector3, latch=True, tcp_nodelay=True)
        self.trgpx_pub = rospy.Publisher('target', Vector3, latch=True, tcp_nodelay=True)
        self.active_pub = rospy.Publisher('active', Bool, latch=True, tcp_nodelay=True)
        self.svg_pub.publish(self.model.svgpath)
        self.srcpx_pub.publish(0,0,0)
        self.trgpx_pub.publish(0,0,0)

        self.timer = rospy.Timer(rospy.Duration(SWITCH_MODE_TIME),
                                  self.switch_conditions)

        rospy.Subscriber("flydra_mainbrain/super_packets",
                         flydra_mainbrain_super_packet,
                         self.on_flydra_mainbrain_super_packets)

    def get_post_pose_msg(self,x,y,z=0):
        msg = Pose()
        msg.position.x = x
        msg.position.y = y
        msg.position.z = z
        msg.orientation.w = 1
        return msg

    def get_hide_post_msg(self):
        return self.get_post_pose_msg(0,0,10)

    def move_point(self, val):
        val,pt = self.model.move_point(val, wrap=SVG_WRAP_PATH)
        self.log.move_ratio = val
        self.moving_ratio = val
        return val

    def switch_conditions(self,event,force=''):
        if force:
            self.condition = force
        else:
            i = CONDITIONS.index(self.condition)
            j = (i+1) % len(CONDITIONS)
            self.condition = CONDITIONS[j]
        self.log.condition = self.condition
        self.drop_lock_on()
        self.p_const = float(self.condition.split('/')[1])
        rospy.loginfo('condition: %s (p=%f)' % (self.condition,self.p_const))

    def get_starfield_velocity_vector(self,t,dt,fly_x,fly_y,fly_z):
        px,py = XFORM.xy_to_pxpy(fly_x,fly_y)

        if is_stepwise_mode(self.condition):
            target = self.model.connect_to_moving_point(p=None,px=px, py=py)
            if target.length < PX_DIST_ADVANCE:
                val = self.move_point(self.moving_ratio + (1.0/CONTROL_RATE))
            else:
                val = self.moving_ratio
        elif is_control_mode(self.condition):
            target = flyflypath.euclid.LineSegment2(
                                flyflypath.euclid.Point2(px,py),
                                flyflypath.euclid.Point2(*XFORM.xy_to_pxpy(self.start_x, self.start_y)))
        else:
            return Vector3(),flyflypath.polyline.ZeroLineSegment2(),False

        #do the control
        msg = Vector3()
        msg.x = target.v.y * +self.p_const
        msg.y = target.v.x * -self.p_const
        msg.z = (fly_z - Z_TARGET) * -5.0

        self.srcpx_pub.publish(px,py,0)
        self.trgpx_pub.publish(target.p2.x,target.p2.y,0)
                
        return msg,target,True

    def run(self):
        rospy.loginfo('running stimulus')
        r = rospy.Rate(CONTROL_RATE)
        while not rospy.is_shutdown():
            with self.trackinglock:
                currently_locked_obj_id = self.currently_locked_obj_id
                fly_x = self.fly.x; fly_y = self.fly.y; fly_z = self.fly.z
                framenumber = self.framenumber

            active = False
            if currently_locked_obj_id is None:
                #FIXME: encourage them to leave the ceiling?
                vec = Vector3()
                target = flyflypath.polyline.ZeroLineSegment2()
            else:
                now = rospy.get_time()
                if now-self.last_seen_time > TIMEOUT:
                    self.drop_lock_on()
                    continue

                if np.isnan(fly_x):
                    #we have a race  - a fly to track with no pose yet
                    #rospy.logwarn('lost tracking, RACE')
                    continue

                self.fly_dist += math.sqrt((fly_x-self.last_fly_x)**2 +
                                           (fly_y-self.last_fly_y)**2 +
                                           (fly_z-self.last_fly_z)**2)
                self.last_fly_x = fly_x; self.last_fly_y = fly_y; self.last_fly_z = fly_z;

                if now-self.last_check_flying_time > FLY_DIST_CHECK_TIME:
                    fly_dist = self.fly_dist
                    self.last_check_flying_time = now
                    self.fly_dist = 0
                    if fly_dist < FLY_DIST_MIN_DIST:
                        self.drop_lock_on()
                        continue

                dt = 1.0/CONTROL_RATE #FIXME, not necessarily...
                vec,target,active = self.get_starfield_velocity_vector(now,dt,fly_x,fly_y,fly_z)

            #no need to write the csv unless we are actually controlling something
            if active:
                self.log.src_x = fly_x; self.log.src_y = fly_y; self.log.src_z = fly_z;
                self.log.stim_x = vec.x; self.log.stim_y = vec.y; self.log.stim_z = vec.z
                self.log.target_x = target.p2.x; self.log.target_y = target.p2.y; self.log.target_z = Z_TARGET
                self.log.active = 1 if active else 0
                self.log.framenumber = framenumber
                self.log.update()

            self.active_pub.publish(active)
            self.starfield_velocity_pub.publish(vec)


            r.sleep()

        rospy.loginfo('%s finished. saved data to %s' % (rospy.get_name(), self.log.close()))

    def is_in_trigger_volume(self,pos):
        c = np.array( (self.start_x, self.start_y) )
        p = np.array( (pos.x, pos.y) )
        dist = np.sqrt(np.sum((c-p)**2))
        radius  = 0.16
        zdist   = 0.4
        if (dist < radius) and (abs(pos.z-Z_TARGET) < zdist):
            return True
        return False

    def on_flydra_mainbrain_super_packets(self,data):
        now = rospy.get_time()
        for packet in data.packets:
            for obj in packet.objects:
                if self.currently_locked_obj_id is not None:
                    if obj.obj_id == self.currently_locked_obj_id:
                        #dont take the lock here, these position updates happen at 200Hz,
                        #no need, off by 1 wont matter
                        self.last_seen_time = now
                        self.fly = obj.position
                        self.framenumber = packet.framenumber
                else:
                    if self.is_in_trigger_volume(obj.position):
                        self.lock_on(obj,packet.framenumber)

    def lock_on(self,obj,framenumber):
        with self.trackinglock:
            self.fly = obj.position
            self.framenumber = framenumber
            self.currently_locked_obj_id = obj.obj_id

            now = rospy.get_time()
            self.first_seen_time = now
            self.last_seen_time = now

            self.last_fly_x = self.fly.x; self.last_fly_y = self.fly.y; self.last_fly_z = self.fly.z;
            self.last_check_flying_time = now
            self.fly_dist = 0

        rospy.loginfo('locked object %d at frame %d at %f,%f,%f' % (
                self.currently_locked_obj_id,self.framenumber,self.fly.x,self.fly.y,self.fly.z))

        #back to the start of the path
        self.move_point(0.0)

        if is_control_nolock_mode(self.condition):
            self.lock_object.publish(IMPOSSIBLE_OBJ_ID_ZERO_POSE)
        else:
            self.lock_object.publish(self.currently_locked_obj_id)

        self.log.lock_object = self.currently_locked_obj_id
        self.log.framenumber = self.framenumber
        self.log.update()

    def drop_lock_on(self):
        with self.trackinglock:
            currently_locked_obj_id = self.currently_locked_obj_id
            self.currently_locked_obj_id = None

            now = rospy.get_time()
            dt = now - self.first_seen_time

        rospy.loginfo('dropping locked object %s (tracked for %s s)' % (currently_locked_obj_id,dt))
        self.lock_object.publish(IMPOSSIBLE_OBJ_ID_ZERO_POSE)

        self.log.lock_object = IMPOSSIBLE_OBJ_ID_ZERO_POSE
        self.log.update()

def main():
    rospy.init_node("followpath")

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-wait', action='store_true', default=False,
                        help="dont't start unless flydra is saving data")
    parser.add_argument('--tmpdir', action='store_true', default=False,
                        help="store logfile in tmpdir")
    argv = rospy.myargv()
    args = parser.parse_args(argv[1:])

    node = Node(
            wait_for_flydra=not args.no_wait,
            use_tmpdir=args.tmpdir)
    return node.run()

if __name__=='__main__':
    main()

