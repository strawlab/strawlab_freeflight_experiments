#!/usr/bin/env python

import os
import numpy as np
import time
import argparse
import threading
import math

PACKAGE='strawlab_freeflight_experiments'
import roslib
roslib.load_manifest(PACKAGE)
import rospy

import freemovr_engine.display_client
import std_msgs.msg
import geometry_msgs.msg
from ros_flydra.msg import flydra_mainbrain_super_packet

import flyflypath.transform
import nodelib.log
import nodelib.visualization

from ros_flydra.constants import IMPOSSIBLE_OBJ_ID, IMPOSSIBLE_OBJ_ID_ZERO_POSE

pkg_dir = roslib.packages.get_pkg_dir(PACKAGE)

CONTROL_RATE = 40.0

TIMEOUT = 0.5

HOLD_COND = "midgray.osg"
OSG_FILE = "L.osgt/0.0,0.0,0.29/0.1,0.1,0.3"

class Logger(nodelib.log.CsvLogger):
    STATE = ("stimulus_filename",)

class Node(object):
    def __init__(self, wait_for_flydra, use_tmpdir, continue_existing, pushover=False):

        self.dsc = freemovr_engine.display_client.DisplayServerProxy.set_stimulus_mode('StimulusOSGFile')

        self.pub_stimulus_scale = rospy.Publisher("model_scale", geometry_msgs.msg.Vector3, latch=True, tcp_nodelay=True)
        self.pub_stimulus_centre = rospy.Publisher("model_pose", geometry_msgs.msg.Pose, latch=True, tcp_nodelay=True)
        self.pub_stimulus = rospy.Publisher("stimulus_filename", std_msgs.msg.String, latch=True, tcp_nodelay=True)

        self.pub_lock_object = rospy.Publisher('lock_object', std_msgs.msg.UInt32, latch=True, tcp_nodelay=True)

        self.log = Logger(wait=wait_for_flydra, use_tmpdir=use_tmpdir, continue_existing=continue_existing)

        if pushover:
            self.pub_pushover = rospy.Publisher('note', std_msgs.msg.String)
        else:
            self.pub_pushover = None
        self.pub_save = rospy.Publisher('save_object', std_msgs.msg.UInt32)

        #protect the traked id and fly position between the time syncronous main loop and the asyn
        #tracking/lockon/off updates
        self.trackinglock = threading.Lock()
        with self.trackinglock:
            self.currently_locked_obj_id = None
            self.fly = geometry_msgs.msg.Vector3()
            self.framenumber = 0
            now = rospy.get_time()
            self.first_seen_time = now
            self.last_seen_time = now

        self.switch_conditions()

        rospy.Subscriber("flydra_mainbrain/super_packets",
                         flydra_mainbrain_super_packet,
                         self.on_flydra_mainbrain_super_packets)

    def switch_conditions(self):

        self.osg_fname,origin,scale = OSG_FILE.split("/")

        xyz = map(float,origin.split(','))
        msg = geometry_msgs.msg.Pose()
        msg.position.x = xyz[0]
        msg.position.y = xyz[1]
        msg.position.z = xyz[2]
        msg.orientation.w = 1.0
        self.pub_stimulus_centre.publish(msg)

        xyz = map(float,scale.split(','))
        msg = geometry_msgs.msg.Vector3(*xyz)
        self.pub_stimulus_scale.publish(msg)

        self.drop_lock_on()

    def run(self):
        rospy.loginfo('running stimulus')
        r = rospy.Rate(CONTROL_RATE)
        while not rospy.is_shutdown():
            with self.trackinglock:
                currently_locked_obj_id = self.currently_locked_obj_id
                fly_x = self.fly.x; fly_y = self.fly.y; fly_z = self.fly.z
                framenumber = self.framenumber

            if currently_locked_obj_id is not None:
                now = rospy.get_time()
                if now-self.last_seen_time > TIMEOUT:
                    self.drop_lock_on()
                    continue

                if (fly_z > 0.95) or (fly_z < 0):
                    self.drop_lock_on()
                    continue

                if np.isnan(fly_x):
                    #we have a race  - a fly to track with no pose yet
                    continue

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

    def update(self, stim):
        self.stimulus_filename = stim

        self.pub_stimulus.publish(stim)
        self.pub_lock_object.publish(self.lock_object)

        self.log.lock_object = self.lock_object
        self.log.framenumber = self.framenumber
        self.log.stimulus_filename = self.stimulus_filename
        self.log.condition = self.stimulus_filename
        self.log.update()

    def lock_on(self,obj,framenumber):
        with self.trackinglock:
            rospy.loginfo('locked object %d at frame %d' % (obj.obj_id,framenumber))
            now = rospy.get_time()
            self.currently_locked_obj_id = obj.obj_id
            self.last_seen_time = now
            self.first_seen_time = now
            self.lock_object = obj.obj_id
            self.framenumber = framenumber

        self.update(self.osg_fname)

    def drop_lock_on(self):
        with self.trackinglock:
            old_id = self.currently_locked_obj_id
            now = rospy.get_time()
            dt = now - self.first_seen_time

            rospy.loginfo('dropping locked object %s (tracked for %.1f)' % (old_id, dt))

            self.currently_locked_obj_id = None

            self.lock_object = IMPOSSIBLE_OBJ_ID
            self.framenumber = 0

        if (dt > 15) and (old_id is not None):
            if self.stimulus_filename != HOLD_COND:
                if self.pub_pushover is not None:
                    self.pub_pushover.publish("Fly %s flew for %.1fs" % (old_id, dt))
                self.pub_save.publish(old_id)

        self.update(HOLD_COND)

def main():
    rospy.init_node("saver")

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
