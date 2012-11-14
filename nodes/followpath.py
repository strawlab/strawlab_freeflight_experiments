#!/usr/bin/env python

import os
import numpy as np
import time
import threading

PACKAGE='strawlab_freeflight_experiments'

import roslib
import roslib.packages
roslib.load_manifest(PACKAGE)

import rospy
import display_client
from std_msgs.msg import String, UInt32, Bool
from geometry_msgs.msg import Vector3
from ros_flydra.msg import flydra_mainbrain_super_packet

import flyflypath.model
import flyflypath.view
import flyflypath.polyline
import nodelib.log

pkg_dir = roslib.packages.get_pkg_dir(PACKAGE)

STARFIELD_TOPIC = 'velocity'

PATH_TO_FOLLOW      = os.path.join(pkg_dir,"data","svgpaths","impspiral3.svg")
CONTROL_RATE        = 20.0      #Hz
MOVING_POINT_TIME   = 25.0      #15s for the target to move along the path (fly move at 0.1m/s)

SWITCH_MODE_TIME    = 3.0*60    #alternate between control and static (i.e. experimental control) seconds

Z_TARGET = 0.5

TIMEOUT = 0.5
IMPOSSIBLE_OBJ_ID = 0
IMPOSSIBLE_OBJ_ID_ZERO_POSE = 0xFFFFFFFF

#FIXME:
SHRINK_SPHERE = 0.8

CONDITIONS = ("follow+control/0.0",
              "follow+stepwise/+0.008",
              "follow+stepwise/+0.006",
              "follow+stepwise/+0.004",
              "follow+stepwise/-0.006")
START_CONDITION = CONDITIONS[2]

def is_constanttime_mode(condition):
    return condition.startswith("follow+constanttime")

def is_stepwise_mode(condition):
    return condition.startswith("follow+stepwise")

def is_static_mode(condition):
    return condition == "static"

def is_control_mode(condition):
    return condition == "follow+control/0.0"

def xy_to_pxpy(x,y):
    #center of svg is at 250,250 - move 0,0 there
    py = (x * +500 * SHRINK_SPHERE) + 250
    px = (y * -500 * SHRINK_SPHERE) + 250
    return px,py

def pxpy_to_xy(px,py):
    y = (px - 250) / -500.0
    x = (py - 250) / +500.0
    return x/SHRINK_SPHERE,y/SHRINK_SPHERE

class Logger(nodelib.log.CsvLogger):
    STATE = ("svg_filename","condition","src_x","src_y","src_z","target_x","target_y","target_z",
                 "stim_x","stim_y","stim_z","move_ratio","active","lock_object","framenumber")

class Node(object):
    def __init__(self):

        #for x,y in ((0.2,0.3),(-0.2,0.4),(0.3,-0.1),(-0.3,-0.45),(0.5,0.5),(0,0)):
        #    px,py = xy_to_pxpy(x,y)

        rospy.init_node("followpath")

        display_client.DisplayServerProxy.set_stimulus_mode(
            'StimulusStarField')

        self.starfield_velocity_pub = rospy.Publisher(STARFIELD_TOPIC, Vector3, latch=True, tcp_nodelay=True)
        self.starfield_velocity_pub.publish(Vector3())
        self.lock_object = rospy.Publisher('lock_object', UInt32, latch=True, tcp_nodelay=True)
        self.lock_object.publish(IMPOSSIBLE_OBJ_ID_ZERO_POSE)

        self.model = flyflypath.model.MovingPointSvgPath(PATH_TO_FOLLOW)
        self.log = Logger(directory="/tmp/")

        startpt = self.model.polyline.p
        self.start_x, self.start_y = pxpy_to_xy(startpt.x,startpt.y)

        self.p_const = 0.0

        #protect the traked id and fly position between the time syncronous main loop and the asyn
        #tracking/lockon/off updates
        self.trackinglock = threading.Lock()
        self.currently_locked_obj_id = None
        self.fly = Vector3()
        now = rospy.get_time()
        self.first_seen_time = now
        self.last_seen_time = now

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
        self.log.framenumber = 0

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

    def move_point(self, val):
        val = self.model.move_point(val)
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
        if is_static_mode(self.condition):
            self.drop_lock_on()
        else:
            self.p_const = float(self.condition.split('/')[1])
        rospy.loginfo('condition: %s (p=%f)' % (self.condition,self.p_const))

    def get_starfield_velocity_vector(self,t,dt,fly_x,fly_y,fly_z):
        px,py = xy_to_pxpy(fly_x,fly_y)

        if is_constanttime_mode(self.condition):
            #advance the point in constant ish time...
            nsteps = MOVING_POINT_TIME/dt
            val = self.move_point(self.moving_ratio + (1.0/nsteps))
        elif is_stepwise_mode(self.condition):
            if (t-self.first_seen_time) > MOVING_POINT_TIME:
                #give up, the fly might have been lost
                val = 1.0
            else:
                dist = self.model.connect_to_moving_point(p=None,px=px, py=py)
                if dist.length < 35:
                    val = self.move_point(self.moving_ratio + (1.0/30))
                else:
                    val = self.moving_ratio
        else:
            rospy.logwarn("condition race")
            return Vector3(),flyflypath.polyline.ZeroLineSegment2(),False

        #finished
        if val == 1.0:
            self.drop_lock_on()
            return Vector3(),flyflypath.polyline.ZeroLineSegment2(),False

        #do the control
        target = self.model.connect_to_moving_point(p=None,px=px, py=py)
        msg = Vector3()
        msg.x = target.v.y * +self.p_const
        msg.y = target.v.x * -self.p_const
        msg.z = (fly_z - Z_TARGET) * -2

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

            active = False
            if currently_locked_obj_id is None:
                #FIXME: encourage them to leave the ceiling?
                vec = Vector3()
                target = flyflypath.polyline.ZeroLineSegment2()
            else:
                now = rospy.get_time()
                if now-self.last_seen_time > TIMEOUT:
                    self.drop_lock_on()
                    self.log.update()
                    continue

                if np.isnan(fly_x):
                    #we have a race  - a fly to track with no pose yet
                    #rospy.logwarn('lost tracking, RACE')
                    continue

                #do the control
                if is_static_mode(self.condition):
                    vec = Vector3()
                    target = flyflypath.polyline.ZeroLineSegment2()
                else:
                    dt = 1.0/CONTROL_RATE #FIXME, not necessarily...
                    vec,target,active = self.get_starfield_velocity_vector(now,dt,fly_x,fly_y,fly_z)

            self.log.src_x = fly_x; self.log.src_y = fly_y; self.log.src_z = fly_z;
            self.log.stim_x = vec.x; self.log.stim_y = vec.y; self.log.stim_z = vec.z
            self.log.target_x = target.p2.x; self.log.target_y = target.p2.y; self.log.target_z = 0.0
            self.log.active = 1 if active else 0

            self.active_pub.publish(active)

            self.log.update()

            self.starfield_velocity_pub.publish(vec)

            r.sleep()

        rospy.loginfo('%s finished. saved data to %s' % (rospy.get_name(), self.log.filename))

    def is_in_trigger_volume(self,pos):
        c = np.array( (self.start_x, self.start_y) )
        p = np.array( (pos.x, pos.y) )
        dist = np.sqrt(np.sum((c-p)**2))
        radius  = 0.16
        zdist   = 0.15
        if (dist < radius) and (abs(pos.z-Z_TARGET) < zdist):
            return True
        return False

    def on_flydra_mainbrain_super_packets(self,data):
        now = rospy.get_time()
        for packet in data.packets:
            for obj in packet.objects:
                if self.currently_locked_obj_id is not None:
                    if obj.obj_id == self.currently_locked_obj_id:
                        self.last_seen_time = now
                        self.fly = obj.position
                else:
                    if self.is_in_trigger_volume(obj.position):
                        if not is_static_mode(self.condition):
                            self.fly = obj.position
                            self.lock_on(obj,packet.framenumber)

    def lock_on(self,obj,framenumber):
        with self.trackinglock:
            self.currently_locked_obj_id = obj.obj_id
    
        rospy.loginfo('locked object %d at frame %d at %f,%f,%f' % (
                self.currently_locked_obj_id,framenumber,self.fly.x,self.fly.y,self.fly.z))
        now = rospy.get_time()
        self.first_seen_time = now
        self.last_seen_time = now
        #back to the start of the path
        self.move_point(0.0)
        self.lock_object.publish(self.currently_locked_obj_id)
        self.log.lock_object = self.currently_locked_obj_id
        self.log.framenumber = framenumber

    def drop_lock_on(self):
        with self.trackinglock:
            currently_locked_obj_id = self.currently_locked_obj_id
            self.currently_locked_obj_id = None

        now = rospy.get_time()
        dt = now - self.first_seen_time
        rospy.loginfo('dropping locked object %s (tracked for %s s)' % (currently_locked_obj_id,dt))
        self.lock_object.publish(IMPOSSIBLE_OBJ_ID_ZERO_POSE)
        self.log.lock_object = IMPOSSIBLE_OBJ_ID_ZERO_POSE
        self.log.framenumber = 0

def main():
    node = Node()
    return node.run()

if __name__=='__main__':
    main()

