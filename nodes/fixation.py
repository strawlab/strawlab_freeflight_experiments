#!/usr/bin/env python

import math
import numpy as np
import threading

PACKAGE='strawlab_freeflight_experiments'
import roslib
roslib.load_manifest(PACKAGE)
import rospy
import display_client
from std_msgs.msg import UInt32
from geometry_msgs.msg import Vector3
from ros_flydra.msg import flydra_mainbrain_super_packet
import rospkg

import nodelib.log

rospack = rospkg.RosPack()
pkg_dir = rospack.get_path(PACKAGE)

STARFIELD_TOPIC = 'velocity'

CONTROL_RATE        = 20.0      #Hz
SWITCH_MODE_TIME    = 3.0*60    #alternate between control and static (i.e. experimental control) seconds

START_N_SEGMENTS    = 8
CYL_RAD             = 0.4
DIST_FROM_WALL      = 0.11

P_CONST_XY          = -2
P_CONST_Z           = -2

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
    STATE = ("condition","start_x","start_y","fly_x","fly_y","lock_object","framenumber")

class Node(object):
    def __init__(self):
        rospy.init_node("fixation")

        display_client.DisplayServerProxy.set_stimulus_mode(
            'StimulusStarField')

        self.starfield_velocity_pub = rospy.Publisher(STARFIELD_TOPIC, Vector3, latch=True, tcp_nodelay=True)
        self.starfield_velocity_pub.publish(Vector3())
        self.lock_object = rospy.Publisher('lock_object', UInt32, latch=True, tcp_nodelay=True)
        self.lock_object.publish(IMPOSSIBLE_OBJ_ID_ZERO_POSE)

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

        self.start_x = self.start_y = 0.0
        self.search_radius = 0.1
        self.search_zdist  = 0.15

        self.switch_conditions(None,force=START_CONDITION)

        self.timer = rospy.Timer(rospy.Duration(SWITCH_MODE_TIME),
                                  self.switch_conditions)

        rospy.Subscriber("flydra_mainbrain_super_packets",
                         flydra_mainbrain_super_packet,
                         self.on_flydra_mainbrain_super_packets)

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
            self.start_x = self.start_y = 0.0
        else:
            self.start_idx = int(self.condition.split('/')[1])
            self.start_x = self.start_coords[self.start_idx][0][0]
            self.start_y = self.start_coords[self.start_idx][0][1]

        #all sub phases start at birth
        self.condition_sub = "birth"

        rospy.loginfo('condition: %s (p=%f)' % (self.condition,self.start_idx))

    def get_starfield_velocity_vector(self,t,dt,fly_x,fly_y,fly_z,target_x,target_y,target_z):
        msg = Vector3()
        msg.x = (fly_x - target_x) * P_CONST_XY
        msg.y = (fly_y - target_y) * P_CONST_XY
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
                elif self.condition_sub == "birth":
                    #all experiments start (birth) by bringing the fly to the start target
                    starfield_velocity = self.get_starfield_velocity_vector(
                                            now,0,
                                            fly_x,fly_y,fly_z,
                                            self.start_x,self.start_y,TARGET_Z)
                else:
                    raise Exception("NOT COMPLETED")

                print starfield_velocity

            self.log.start_x = self.start_x; self.log.start_y = self.start_y; self.log.start_z = TARGET_Z
            self.log.fly_x = fly_x; self.log.fly_y = fly_y; self.log.fly_z = fly_z;
            self.log.update()

            self.starfield_velocity_pub.publish(starfield_velocity)

            r.sleep()

    def is_in_trigger_volume(self,pos):
        c = np.array( (self.start_x, self.start_y) )
        p = np.array( (pos.x, pos.y) )
        dist = np.sqrt(np.sum((c-p)**2))
        if (dist < self.search_radius) and (abs(pos.z-TARGET_Z) < self.search_zdist):
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

