#!/usr/bin/env python

import os
import numpy as np
import time
import argparse

PACKAGE='strawlab_freeflight_experiments'
import roslib
roslib.load_manifest(PACKAGE)
import rospy
import flyvr.display_client as display_client
from std_msgs.msg import String, UInt32
from geometry_msgs.msg import Vector3, Pose
from ros_flydra.msg import flydra_mainbrain_super_packet

import nodelib.log
from ros_flydra.constants import IMPOSSIBLE_OBJ_ID, IMPOSSIBLE_OBJ_ID_ZERO_POSE

pkg_dir = roslib.packages.get_pkg_dir(PACKAGE)

CONDITIONS = ("midgray.osg/+0.0/+0.000",
              "checkerboard.png.osg/+0.0/+0.000")
START_CONDITION = CONDITIONS[0]

TIMEOUT = 0.5

class Logger(nodelib.log.CsvLogger):
    STATE = ("stimulus_filename",)

class Node(object):
    def __init__(self, wait_for_flydra, use_tmpdir):

        self._pub_stim_mode = display_client.DisplayServerProxy.set_stimulus_mode(
            'StimulusOSGFile')

        self.log = Logger(wait=wait_for_flydra, use_tmpdir=use_tmpdir)

        self.pub_stimulus = rospy.Publisher('stimulus_filename', String, latch=True, tcp_nodelay=True)
        self.pub_lock_object = rospy.Publisher('lock_object', UInt32, latch=True, tcp_nodelay=True)
        self.pub_model_pose = rospy.Publisher('model_pose', Pose, latch=True, tcp_nodelay=True)

        self.currently_locked_obj_id = None
        self.timer = rospy.Timer(rospy.Duration(5*60), # switch every 5 minutes
                                  self.switch_conditions)
        self.switch_conditions(None,force=START_CONDITION)

        rospy.Subscriber("flydra_mainbrain/super_packets",
                         flydra_mainbrain_super_packet,
                         self.on_flydra_mainbrain_super_packets)

    def get_model_pose_msg(self):
        msg = Pose()
        msg.position.x = self.x0
        msg.position.y = self.y0
        msg.position.z = 0.0
        msg.orientation.w = 1
        return msg

    def switch_conditions(self,event,force=''):
        if force:
            self.condition = force
        else:
            i = CONDITIONS.index(self.condition)
            j = (i+1) % len(CONDITIONS)
            self.condition = CONDITIONS[j]

        self.stimulus_filename = self.condition.split('/')[0]
        self.x0,self.y0 = map(float,self.condition.split('/')[1:])

        self.log.condition = self.condition
        rospy.loginfo('condition: %s (%f,%f)' % (self.condition,self.x0,self.y0))
        self.drop_lock_on()

    def run(self):
        rospy.loginfo('running stimulus')
        r = rospy.Rate(10) # 10hz
        while not rospy.is_shutdown():
            if self.currently_locked_obj_id is not None:
                now = rospy.get_time()
                if now-self.last_seen_time > TIMEOUT:
                    self.drop_lock_on()
            r.sleep()
        rospy.loginfo('%s finished. saved data to %s' % (rospy.get_name(), self.log.close()))

    def is_in_trigger_volume(self,pos,expanded=False):
        c = np.array( (self.x0,self.y0) )
        p = np.array( (pos.x, pos.y) )
        dist = np.sqrt(np.sum((c-p)**2))
        radius = 0.16
        zdist = 0.4
        if expanded:
            radius += 0.05
            zdist += 0.05

        if (dist < radius) and (abs(pos.z-0.5) < zdist):
            return True
        return False

    def on_flydra_mainbrain_super_packets(self,data):
        now = rospy.get_time()
        for packet in data.packets:
            for obj in packet.objects:
                if self.currently_locked_obj_id is not None:
                    # if not self.is_in_trigger_volume(obj.position,expanded=True):
                    #     self.drop_lock_on()
                    if obj.obj_id == self.currently_locked_obj_id:
                        self.last_seen_time = now
                    #print obj
                else:
                    if self.is_in_trigger_volume(obj.position):
                        #print '*'*200
                        #print obj
                        self.lock_on(obj,packet.framenumber)

    def update(self):
        self.log.update()
        self.pub_lock_object.publish( self.log.lock_object )
        self.pub_stimulus.publish( self.stimulus_filename )
        self.pub_model_pose.publish( self.get_model_pose_msg() )

    def lock_on(self,obj,framenumber):
        rospy.loginfo('locked object %d at frame %d' % (obj.obj_id,framenumber))
        now = rospy.get_time()
        self.currently_locked_obj_id = obj.obj_id
        self.last_seen_time = now
        self.log.lock_object = obj.obj_id
        self.log.framenumber = framenumber
        self.update()

    def drop_lock_on(self):
        rospy.loginfo('dropping locked object %s' % self.currently_locked_obj_id)
        self.currently_locked_obj_id = None
        self.log.lock_object = IMPOSSIBLE_OBJ_ID
        self.log.framenumber = 0
        self.update()

def main():
    rospy.init_node("confinement")

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
