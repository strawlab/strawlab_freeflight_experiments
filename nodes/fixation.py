#!/usr/bin/env python

import math
import numpy as np
import threading

PACKAGE='strawlab_freeflight_experiments'

import roslib
import roslib.packages
roslib.load_manifest(PACKAGE)

import rospy
import display_client
from std_msgs.msg import UInt32
from geometry_msgs.msg import Vector3, Pose
from ros_flydra.msg import flydra_mainbrain_super_packet
import rospkg

import nodelib.log

pkg_dir = roslib.packages.get_pkg_dir(PACKAGE)

STARFIELD_TOPIC = 'velocity'
POST_TOPIC      = 'model_pose'


CONTROL_RATE        = 20.0      #Hz
SWITCH_MODE_TIME    = 3.0*60    #alternate between control and static (i.e. experimental control) seconds

START_N_SEGMENTS    = 3
CYL_RAD             = 0.4
DIST_FROM_WALL      = 0.15

P_X                 = -3
P_Y                 = -3
P_CONST_Z           = -3

TARGET_Z            = 0.5

TIMEOUT = 0.5
IMPOSSIBLE_OBJ_ID = 0
IMPOSSIBLE_OBJ_ID_ZERO_POSE = 0xFFFFFFFF

CONDITIONS = ["static"]
CONDITIONS.extend( "stripe_fixate/%d" % i for i in range(START_N_SEGMENTS) )
START_CONDITION = CONDITIONS[1]

SUB_CONDITIONS = ["birth","experiment","death"]

def is_stripe_fixate(condition):
    return condition.startswith("stripe_fixate")

def is_static_mode(condition):
    return condition == "static"

class Logger(nodelib.log.CsvLogger):
    STATE = ("condition","condition_sub","trg_x","trg_y","fly_x","fly_y","fly_z","lock_object","framenumber")

class Node(object):
    def __init__(self):
        rospy.init_node("fixation")

        display_client.DisplayServerProxy.set_stimulus_mode(
            'StimulusStarFieldAndModel')

        self.starfield_velocity_pub = rospy.Publisher(STARFIELD_TOPIC, Vector3, latch=True, tcp_nodelay=True)
        self.starfield_velocity_pub.publish(Vector3())
        self.starfield_post_pub = rospy.Publisher(POST_TOPIC, Pose, latch=True, tcp_nodelay=True)
        self.starfield_post_pub.publish(self.get_hide_post_msg())
        self.lock_object = rospy.Publisher('lock_object', UInt32, latch=True, tcp_nodelay=True)
        self.lock_object.publish(IMPOSSIBLE_OBJ_ID_ZERO_POSE)

        #self.log = Logger(directory="/tmp/")
        self.log = Logger()

        #protect the traked id and fly position between the time syncronous main loop and the asyn
        #tracking/lockon/off updates
        self.trackinglock = threading.Lock()
        self.currently_locked_obj_id = None
        self.fly = Vector3()
        now = rospy.get_time()
        self.first_seen_time = now
        self.last_seen_time = now

        #init the log to zero (we publish immediately)
        self.log.fly_x = self.fly.x; self.log.fly_y = self.fly.y; self.log.fly_z = self.fly.z;
        self.log.lock_object = IMPOSSIBLE_OBJ_ID_ZERO_POSE
        self.log.framenumber = 0

        #calculate the starting targets for the starfield controller, and the point
        #opposing them in the cylinder (for the stripe fixation)
        res = []
        for ang in np.linspace(0,2*np.pi,START_N_SEGMENTS+1):
            x = math.cos(ang) * (CYL_RAD - DIST_FROM_WALL)
            y = math.sin(ang) * (CYL_RAD - DIST_FROM_WALL)
            res.append( [(x,y),(-x,-y)] )
        self.start_coords = res
        self.start_idx = 0

        self.trg_x = self.trg_y = 0.0
        self.post_x = self.post_y = 0.0

        self.fly_pub = rospy.Publisher('~fly', Vector3)
        self.trg_pub = rospy.Publisher('~target', Vector3)
        self.post_pub = rospy.Publisher('~post', Vector3)

        self.search_radius_birth = 0.15
        self.search_radius_death = 0.08

        self.search_zdist  = 0.15

        self.switch_conditions(None,force=START_CONDITION)

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

    def switch_sub_conditions(self,condition_sub):
        self.switch_conditions(None,self.condition,condition_sub)

    def switch_conditions(self,event,force='',condition_sub="birth"):
        if force:
            self.condition = force
        else:
            i = CONDITIONS.index(self.condition)
            j = (i+1) % len(CONDITIONS)
            self.condition = CONDITIONS[j]
        self.log.condition = self.condition
        if is_static_mode(self.condition):
            self.drop_lock_on()
            self.trg_x = self.trg_y = 0.0
            self.post_x = self.post_y = 0.0
            self.p_x = self.p_y = 0.0
        else:
            cond,start_idx = self.condition.split('/')
            self.start_idx = int(start_idx)
            self.p_x = float(P_X)
            self.p_y = float(P_Y)
            self.trg_x, self.trg_y = self.start_coords[self.start_idx][0]
            self.post_x, self.post_y = self.start_coords[self.start_idx][1]

        #all sub phases start at birth
        self.condition_sub = condition_sub
        self.log.condition_sub = self.condition_sub

        rospy.loginfo('condition: %s:%s' % (self.condition,self.condition_sub))

    def get_starfield_velocity_vector(self,t,dt,fly_x,fly_y,fly_z,target_x,target_y,target_z):
        msg = Vector3()
        msg.x = (fly_x - target_x) * self.p_x
        msg.y = (fly_y - target_y) * self.p_y
        msg.z = (fly_z - target_z) * P_CONST_Z
        return msg

    def run(self):
        rospy.loginfo('running stimulus')
        r = rospy.Rate(CONTROL_RATE)
        while not rospy.is_shutdown():
            with self.trackinglock:
                currently_locked_obj_id = self.currently_locked_obj_id
                fly_x = self.fly.x; fly_y = self.fly.y; fly_z = self.fly.z

            if currently_locked_obj_id is None:
                starfield_velocity = Vector3() #zero velocity in x,y,z
                starfield_post = self.get_hide_post_msg()
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
                    starfield_velocity = Vector3() #zero velocity in x,y,z
                    starfield_post = self.get_hide_post_msg()
                elif self.condition_sub == "birth":
                    #unless already there, all experiments start (birth) by bringing the fly to the
                    #start target
                    if self.is_in_trigger_volume(fly_x,fly_y,fly_z,self.trg_x,self.trg_y,TARGET_Z,self.search_radius_death):
                        self.switch_sub_conditions("experiment")
                        starfield_velocity = Vector3()
                        starfield_post = self.get_hide_post_msg()
                    else:
                        starfield_velocity = self.get_starfield_velocity_vector(
                                                now,0,
                                                fly_x,fly_y,fly_z,
                                                self.trg_x,self.trg_y,TARGET_Z)
                elif self.condition_sub == "experiment":
                    #ignore z or also control z?
                    if self.is_in_trigger_volume(fly_x,fly_y,0,self.post_x,self.post_y,0,self.search_radius_death):
                        self.switch_sub_conditions("birth")
                        self.drop_lock_on()
                    else:
                        starfield_velocity = Vector3()
                        starfield_post = self.get_post_pose_msg(self.post_x,self.post_y)
                else:
                    raise Exception("NOT COMPLETED")

            self.log.trg_x = self.trg_x; self.log.trg_y = self.trg_y; self.log.trg_z = TARGET_Z
            self.log.fly_x = fly_x; self.log.fly_y = fly_y; self.log.fly_z = fly_z;
            self.log.update()

            self.fly_pub.publish(fly_x,fly_y,fly_z)
            self.trg_pub.publish(self.trg_x,self.trg_y,TARGET_Z)
            self.post_pub.publish(self.post_x,self.post_y,0)

            self.starfield_velocity_pub.publish(starfield_velocity)
            self.starfield_post_pub.publish(starfield_post)

            r.sleep()

        rospy.loginfo('%s finished. saved data to %s' % (rospy.get_name(), self.log.filename))

    def is_in_trigger_volume(self,fly_x,fly_y,fly_z,x,y,z,search_radius):
        c = np.array( (x, y) )
        p = np.array( (fly_x, fly_y) )
        dist = np.sqrt(np.sum((c-p)**2))
        if (dist < search_radius) and (abs(fly_z-z) < self.search_zdist):
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
                    if self.condition_sub in ("birth",):
                        #when birthing flies taken them from anywhere in the middle of the
                        #arena and move them to the target
                        if self.is_in_trigger_volume(obj.position.x,obj.position.y,obj.position.z,
                                                     0.0, 0.0, TARGET_Z,
                                                     self.search_radius_birth):
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

