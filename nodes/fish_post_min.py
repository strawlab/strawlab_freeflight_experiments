#!/usr/bin/env python
"""
updated for fish

TODO: 
- add geometric filter for fishbowl
	start_radius and stop_radius from center of shoere that is at 0.12
"""
import math
import numpy as np
import threading
import argparse
import os.path
import time
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


CONTROL_RATE        = 80.0      #Hz

HOLD_COND = "fish_bowl_ico_6.osgt"

FLY_DIST_CHECK_TIME = 5000.0
FLY_DIST_MIN_DIST   = 0.03



#####
# checked

TIMEOUT             = 0.5

class Node(nodelib.node.Experiment):
    def __init__(self, args):
        super(Node, self).__init__(args=args,
                                   state=("stimulus_filename"))

        self._pub_stim_mode = display_client.DisplayServerProxy.set_stimulus_mode(
            'StimulusOSGFile')

        self.pub_stimulus = rospy.Publisher('stimulus_filename', String, latch=True, tcp_nodelay=True)

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

        self.ack_pub = rospy.Publisher("active", Bool)

        self.switch_conditions()
        rospy.Subscriber("flydra_mainbrain/super_packets",
                         flydra_mainbrain_super_packet,
                         self.on_flydra_mainbrain_super_packets)

    def switch_conditions(self):
        self.drop_lock_on()
        self.stimulus_filename = str(self.condition['stimulus_filename'])
        self.log.stimulus_filename = self.stimulus_filename
        rospy.loginfo('NEW CONDITION: %s' % (self.condition.name))


    def run(self):
        rospy.loginfo('running stimulus')
        r = rospy.Rate(CONTROL_RATE)
        while not rospy.is_shutdown():
	    now = rospy.get_time()
            with self.trackinglock:
                currently_locked_obj_id = self.currently_locked_obj_id
		fly = self.fly
                fly_x = self.fly.x; fly_y = self.fly.y; fly_z = self.fly.z
                fly_vx = self.flyv.x; fly_vy = self.flyv.y; fly_vz = self.flyv.z
                framenumber = self.framenumber
	    
  
            if currently_locked_obj_id is None:
                active = False
            else:
                
                if now-self.last_seen_time > TIMEOUT:
                    self.drop_lock_on('timeout')
                    rospy.loginfo('TIMEOUT: time since last seen >%.1fs' % (TIMEOUT))
                    continue

                #if not self._is_in_bowl_for_start(fly):
                #    self.drop_lock_on('left volume')
                #    continue

                if np.isnan(fly_x):
                    #we have a race  - a fly to track with no pose yet
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

                #px,py = XFORM.xy_to_pxpy(fly_x,fly_y)
                #self.src_pub.publish(px,py,fly_z)


            #new combine needs data recorded at the framerate
            self.log.framenumber = framenumber
            self.log.update()

            self.ack_pub.publish(active)

            r.sleep()

        rospy.loginfo('%s finished. saved data to %s' % (rospy.get_name(), self.log.close()))


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
                    self.lock_on(obj,packet.framenumber)

    def update(self):
        self.log.update()
        self.pub_lock_object.publish( self.log.lock_object )

    def lock_on(self,obj,framenumber):
	print 'lock on'
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
	    time.sleep(0.1)

        self.pub_stimulus.publish( HOLD_COND )
        self.update()        

def main():
    rospy.init_node("confinement")
    parser, args = nodelib.node.get_and_parse_commandline()
    node = Node(args)
    return node.run()

if __name__=='__main__':
    main()
