#!/usr/bin/env python

# This stimulus is intendedto just show a dark or bright stationary background
# no locking should be done. For this reason stationary .osgt scenes were adapted
# they are switched at adefigned rate
#
# TODO: get the locking out and the logging right
#
#
#
#
#
#
import math
import numpy as np
import threading
import argparse
import os.path

PACKAGE='strawlab_freeflight_experiments'

import roslib
import roslib.packages
roslib.load_manifest(PACKAGE)

import rospy
import flyvr.display_client as display_client
from std_msgs.msg import UInt32, Bool, Float32, String
import std_msgs.msg
from geometry_msgs.msg import Vector3, Pose
from ros_flydra.msg import flydra_mainbrain_super_packet

import flyflypath.model
import flyflypath.transform
import nodelib.log
import strawlab_freeflight_experiments.replay as sfe_replay

from strawlab_freeflight_experiments.topics import *

pkg_dir = roslib.packages.get_pkg_dir(PACKAGE)

CONTROL_RATE        = 80.0      #Hz
SWITCH_MODE_TIME    = 5 #10.0*60    #alternate between control and static (i.e. experimental control) seconds

ADVANCE_RATIO       = 1/100.0   #### ????? ####

FLY_DIST_CHECK_TIME = 1       # time interval in seconds to check fly movement
FLY_DIST_MIN_DIST   = 0.001   # swimm more than 0.5 cm/sec # minimum distance fly must move in above interval to not be ignored

# start volume defined as cube
X_MIN = -0.15
X_MAX =  0.15
Y_MIN = -0.15
Y_MAX =  0.15
Z_MIN =  -0.09
Z_MAX =  0.01





# time interval in seconds to check fly movement after lock on
FLY_HEIGHT_CHECK_TIME = 2 #5 was good #0.5       

# z range for fly tracking (dropped outside)
# this range is only tested after the fly has been tracked for FLY_HEIGHT_CHECK_TIME seconds
Z_MINIMUM = -0.10 # 0.01 max
Z_MAXIMUM = 0.01 # 0.38 max

# y range for fly tracking (dropped outside)
# this range is only tested after the fly has been tracked for FLY_HEIGHT_CHECK_TIME seconds
Y_MINIMUM = -0.18
Y_MAXIMUM = 0.18



TIMEOUT             = 5
IMPOSSIBLE_OBJ_ID   = 0             #### ????? ####

PI = np.pi
TAU= 2*PI



#CONDITION =  .osg Filename
#		duration to test
# 


CONDITIONS = [ 'cylinder_white.osgt','cylinder_blak.osgt']


START_CONDITION = CONDITIONS[0]
#If there is a considerable flight in these conditions then a pushover
#message is sent and a video recorded
COOL_CONDITIONS = set(CONDITIONS[0:3])
MAX_COOL = 10

XFORM = flyflypath.transform.SVGTransform()

class Logger(nodelib.log.CsvLogger):
    STATE = ("stimulus_filename",)

class Node(object):
    def __init__(self, wait_for_flydra, use_tmpdir, continue_existing):

        self.dsc = display_client.DisplayServerProxy.set_stimulus_mode('StimulusOSGFile')  


	self.pub_stimulus = rospy.Publisher("stimulus_filename", std_msgs.msg.String, latch=True, tcp_nodelay=True)

        self.pub_pushover = rospy.Publisher('note', String)
        self.pub_save = rospy.Publisher('save_object', UInt32)

        self.pub_lock_object = rospy.Publisher('lock_object', UInt32, latch=True, tcp_nodelay=True)
        self.pub_lock_object.publish(IMPOSSIBLE_OBJ_ID)

        self.log = Logger(wait=wait_for_flydra, use_tmpdir=use_tmpdir, continue_existing=continue_existing)

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
            self.model = None

            self.ratio_total = 0

            self.replay_rotation = sfe_replay.ReplayStimulus(default=0.0)
            self.replay_z = sfe_replay.ReplayStimulus(default=0.0)

            self.blacklist = {}

        self.switch_conditions(None,force=START_CONDITION)

        self.timer = rospy.Timer(rospy.Duration(SWITCH_MODE_TIME),
                                  self.switch_conditions)
	self.ack_pub = rospy.Publisher("active", Bool)
        rospy.Subscriber("flydra_mainbrain/super_packets",
                         flydra_mainbrain_super_packet,
                         self.on_flydra_mainbrain_super_packets)

    @property
    def is_replay_experiment_rotation(self):
        return np.isnan(self.p_const)
    @property
    def is_replay_experiment_z(self):
        return np.isnan(self.v_gain)



    def switch_conditions(self,event,force=''):
        if force:
            self.condition = force
        else:
            i = CONDITIONS.index(self.condition)
            j = (i+1) % len(CONDITIONS)
            self.condition = CONDITIONS[j]
        
	self.log.condition = self.condition

        self.drop_lock_on()

	self.pub_stimulus.publish(self.condition)
        

        rospy.loginfo('condition: %s' % (self.condition))



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
                    self.drop_lock_on()
                    rospy.loginfo('TIMEOUT: time since last seen >%.1fs' % (TIMEOUT))
                    continue

                # check if fly is in an acceptable z range after a given interval after lock_on
                if ((now - self.first_seen_time) > FLY_HEIGHT_CHECK_TIME) and ((fly_z > Z_MAXIMUM) or (fly_z < Z_MINIMUM)):
                    self.drop_lock_on()
                    if (fly_z > Z_MAXIMUM):
                        rospy.loginfo('MAXIMUM: too high (Z = %.2f > Z_MAXIMUM %.2f )' % (fly_z, Z_MAXIMUM))
                    else:
                        rospy.loginfo('MINIMUM: too low (Z = %.2f < Z_MINIMUM %.2f )' % (fly_z, Z_MINIMUM))
                    continue

                if np.isnan(fly_x):
                    #we have a race  - a fly to track with no pose yet
                    continue

                active = True

                # check if fly is in an acceptable y range after a given interval after lock_on
                if ((now - self.first_seen_time) > FLY_HEIGHT_CHECK_TIME) and ((fly_y > Y_MAXIMUM) or (fly_y < Y_MINIMUM)):
                    self.drop_lock_on()
                    if (fly_y > Y_MAXIMUM):
                        rospy.loginfo('WALL: Wall (Y = %.2f > Y_MAXIMUM %.2f )' % (fly_y, Y_MAXIMUM))
                    else:
                        rospy.loginfo('WALL: Wall (Y = %.2f < Z_MINIMUM %.2f )' % (fly_y, Y_MINIMUM))
                    continue   

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
                        self.drop_lock_on()
                        rospy.loginfo('SLOW: too slow (%.3f < %.3f m/s)' % (fly_dist/FLY_DIST_CHECK_TIME, FLY_DIST_MIN_DIST/FLY_DIST_CHECK_TIME))
                        continue

                self.log.framenumber = framenumber

                self.log.update()

            self.ack_pub.publish(active)

            r.sleep()

        rospy.loginfo('%s finished. saved data to %s' % (rospy.get_name(), self.log.close()))

    def should_lock_on(self, obj):
        if obj.obj_id in self.blacklist:
            return False

        pos = obj.position 
        if ((pos.x>X_MIN) and (pos.x<X_MAX) and (pos.y>Y_MIN) and (pos.y<Y_MAX) and (pos.z>Z_MIN) and (pos.z<Z_MAX)):
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
                        self.flyv = obj.velocity
                        self.framenumber = packet.framenumber
                else:
                    if self.should_lock_on(obj):
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

            self.ratio_total = 0

            self.replay_rotation.reset()
            self.replay_z.reset()

        self.update()

    def drop_lock_on(self, blacklist=False):
        with self.trackinglock:
            old_id = self.currently_locked_obj_id
            now = rospy.get_time()
            dt = now - self.first_seen_time

            rospy.loginfo('dropping locked object %s (tracked for %.1f, %.1f loops)' % (old_id, dt, self.ratio_total))

            self.currently_locked_obj_id = None

            self.log.lock_object = IMPOSSIBLE_OBJ_ID
            self.log.framenumber = 0

            self.log.ratio = 0

            if blacklist:
                self.blacklist[old_id] = True




        self.update()

def main():
    rospy.init_node("translation")

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-wait', action='store_true', default=False,
                        help="dont't start unless flydra is saving data")
    parser.add_argument('--tmpdir', action='store_true', default=False,
                        help="store logfile in tmpdir")
    parser.add_argument('--continue-existing', type=str, default=None,
                        help="path to a logfile to continue")
    argv = rospy.myargv()
    args = parser.parse_args(argv[1:])

    node = Node(
            wait_for_flydra=not args.no_wait,
            use_tmpdir=args.tmpdir,
            continue_existing=args.continue_existing)
    return node.run()

if __name__=='__main__':
    main()

