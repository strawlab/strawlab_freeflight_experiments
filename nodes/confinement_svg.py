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
import display_client
from std_msgs.msg import String, UInt32, Bool, Float32
from geometry_msgs.msg import Vector3, Pose, PoseArray, Point
from ros_flydra.msg import flydra_mainbrain_super_packet
import rospkg

import flyflypath.transform
import nodelib.log
from ros_flydra.constants import IMPOSSIBLE_OBJ_ID, IMPOSSIBLE_OBJ_ID_ZERO_POSE

pkg_dir = roslib.packages.get_pkg_dir(PACKAGE)

HOLD_COND = "midgray.osg"

CONDITIONS = ("midgray.osg/+0.0/+0.0/0",
#              "lboxmed.svg.osg/+0.0/+0.0/1",
#              "lboxmed.svg.osg/+0.0/+0.0/5",
#              "lboxmed.svg.osg/+0.0/+0.0/25",
#              "lboxmed.svg.osg/+0.0/+0.0/125",
#              "lboxmed.svg.osg/+0.0/+0.0/625"
              "lboxmed.svg.osg/+0.0/+0.0/0",
              "lboxmed.svg.osg/+0.0/+0.0/-1",
)
START_CONDITION = CONDITIONS[1]

CONTROL_RATE = 40.0

FLY_DIST_CHECK_TIME = 5.0
FLY_DIST_MIN_DIST = 0.2

START_RADIUS    = 0.12
START_ZDIST     = 0.4
START_Z         = 0.5

TIMEOUT = 0.5

XFORM = flyflypath.transform.SVGTransform()

class Logger(nodelib.log.CsvLogger):
    STATE = ("stimulus_filename","confinement_condition","lock_object","framenumber","startr")

class Node(object):
    def __init__(self, wait_for_flydra, use_tmpdir):

        self._pub_stim_mode = display_client.DisplayServerProxy.set_stimulus_mode(
            'StimulusOSGFile')

        self.log = Logger(wait=wait_for_flydra, use_tmpdir=use_tmpdir)

        self.pub_stimulus = rospy.Publisher('stimulus_filename', String, latch=True, tcp_nodelay=True)
        self.pub_lock_object = rospy.Publisher('lock_object', UInt32, latch=True, tcp_nodelay=True)
        self.pub_lag = rospy.Publisher("extra_lag_msec", Float32, latch=True, tcp_nodelay=True)

        #publish for the follow_path monitor
        self.svg_pub = rospy.Publisher('svg_filename', String, latch=True, tcp_nodelay=True)
        self.srcpx_pub = rospy.Publisher('source', Vector3, latch=False, tcp_nodelay=True)
        self.active_pub = rospy.Publisher('active', Bool, latch=True, tcp_nodelay=True)
        self.trigarea_pub = rospy.Publisher('trigger_area', PoseArray, latch=True, tcp_nodelay=True)

        #protect the traked id and fly position between the time syncronous main loop and the asyn
        #tracking/lockon/off updates
        self.trackinglock = threading.Lock()
        with self.trackinglock:
            self.currently_locked_obj_id = None
            self.fly = Vector3()
            self.framenumber = 0
            now = rospy.get_time()
            self.first_seen_time = now
            self.last_seen_time = now
            self.last_fly_x = self.fly.x; self.last_fly_y = self.fly.y; self.last_fly_z = self.fly.z;
            self.last_check_flying_time = now
            self.fly_dist = 0

        self.timer = rospy.Timer(rospy.Duration(5*60), # switch every 5 minutes
                                  self.switch_conditions)
        self.switch_conditions(None,force=START_CONDITION)

        self.srcpx_pub.publish(0,0,0)
        self.active_pub.publish(False)
        self.trigarea_pub.publish(self._get_trigger_volume_posearray())

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
        self.x0,self.y0 = map(float,self.condition.split('/')[1:-1])

        lag = float(self.condition.split('/')[-1])
        self.pub_lag.publish(lag)

        svg_filename = os.path.join(pkg_dir,"data","svgpaths",self.stimulus_filename[:-4])
        self.svg_pub.publish(svg_filename)

        self.log.confinement_condition = self.condition
        self.log.startr = START_RADIUS
        rospy.loginfo('confinement condition: %s (%f,%f)' % (self.condition,self.x0,self.y0))
        self.drop_lock_on()

    def run(self):
        rospy.loginfo('running stimulus')
        r = rospy.Rate(CONTROL_RATE)
        while not rospy.is_shutdown():
            with self.trackinglock:
                currently_locked_obj_id = self.currently_locked_obj_id
                fly_x = self.fly.x; fly_y = self.fly.y; fly_z = self.fly.z
                framenumber = self.framenumber

            if currently_locked_obj_id is None:
                active = False
            else:
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

                active = True

                #distance accounting, give up on fly if it is not moving
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

                px,py = XFORM.xy_to_pxpy(fly_x,fly_y)
                self.srcpx_pub.publish(px,py,0)

            #don't need to record anything at the control rate
            #self.log.framenumber = framenumber
            #self.log.update()


            self.active_pub.publish(active)

            r.sleep()

        rospy.loginfo('%s finished. saved data to %s' % (rospy.get_name(), self.log.close()))

    def _get_trigger_volume_posearray(self):

        def _xy_on_circle(radius, ox, oy, steps=16):
            angleStep = 2 * math.pi / steps
            for a in range(0, steps):
                x = math.sin(a * angleStep) * radius + ox
                y = math.cos(a * angleStep) * radius + oy
                yield x, y

        pxpy = [XFORM.xy_to_pxpy(*v) for v in _xy_on_circle(START_RADIUS,self.x0,self.y0)]
        poses = [Pose(position=Point(px,py,0)) for px,py in pxpy]
        return PoseArray(poses=poses)

    def is_in_trigger_volume(self,pos):
        c = np.array( (self.x0,self.y0) )
        p = np.array( (pos.x, pos.y) )
        dist = np.sqrt(np.sum((c-p)**2))
        if (dist < START_RADIUS) and (abs(pos.z-START_Z) < START_ZDIST):
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
                        self.framenumber = packet.framenumber
                else:
                    if self.is_in_trigger_volume(obj.position):
                        self.lock_on(obj,packet.framenumber)

    def update(self):
        self.log.update()
        self.pub_lock_object.publish( self.log.lock_object )

    def lock_on(self,obj,framenumber):
        rospy.loginfo('locked object %d at frame %d' % (obj.obj_id,framenumber))
        now = rospy.get_time()
        self.currently_locked_obj_id = obj.obj_id
        self.last_seen_time = now
        self.log.lock_object = obj.obj_id
        self.log.framenumber = framenumber
        self.pub_stimulus.publish( self.stimulus_filename )
        self.update()

    def drop_lock_on(self):
        rospy.loginfo('dropping locked object %s' % self.currently_locked_obj_id)
        self.currently_locked_obj_id = None
        self.log.lock_object = IMPOSSIBLE_OBJ_ID
        self.log.framenumber = 0
        self.pub_stimulus.publish( HOLD_COND )
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