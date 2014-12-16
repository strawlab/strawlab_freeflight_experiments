#!/usr/bin/env python

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
from geometry_msgs.msg import Vector3, Pose, Polygon
from ros_flydra.msg import flydra_mainbrain_super_packet

import flyflypath.transform
import nodelib.log
import nodelib.visualization

from ros_flydra.constants import IMPOSSIBLE_OBJ_ID

pkg_dir = roslib.packages.get_pkg_dir(PACKAGE)

CONTROL_RATE        = 40.0      #Hz
SWITCH_MODE_TIME    = 5.0*60    #alternate between control and static (i.e. experimental control) seconds

HOLD_COND = "midgray.osg"

#CONDITION = "stimulus_filename,x0,y0,lag (ms, -ve = infinite)"
CONDITIONS = [
              "midgray.osg/+0.0/+0.0/0",
              "lboxmed.svg.osg/+0.0/+0.0/0",
#              "lboxmed.svg.osg/+0.0/+0.0/0",
#              "lboxmed.svg.osg/+0.0/+0.0/0",
#              "lboxmed.svg.osg/+0.0/+0.0/1",
#              "lboxmed.svg.osg/+0.0/+0.0/5",
#              "lboxmed.svg.osg/+0.0/+0.0/25",
#              "lboxmed.svg.osg/+0.0/+0.0/125",
#              "lboxmed.svg.osg/+0.0/+0.0/625",
#              "lboxmed.svg.osg/+0.0/+0.0/-1",
]
START_CONDITION = 1
COOL_CONDITIONS = set(["lboxmed.svg.osg/+0.0/+0.0/0"])#set()


FLY_DIST_CHECK_TIME = 5.0
FLY_DIST_MIN_DIST   = 0.2

START_RADIUS    = 0.12
START_ZDIST     = 0.4
START_Z         = 0.5

# z range for fly tracking (dropped outside)
Z_MINIMUM = 0.00
Z_MAXIMUM = 0.95

TIMEOUT             = 0.5

XFORM = flyflypath.transform.SVGTransform()

class Logger(nodelib.log.CsvLogger):
    STATE = ("stimulus_filename","startr")

class Node(object):
    def __init__(self, wait_for_flydra, use_tmpdir, continue_existing):

        self._pub_stim_mode = display_client.DisplayServerProxy.set_stimulus_mode(
            'StimulusOSGFile')

        self.pub_stimulus = rospy.Publisher('stimulus_filename', String, latch=True, tcp_nodelay=True)
        self.pub_lag = rospy.Publisher('extra_lag_msec', Float32, latch=True, tcp_nodelay=True)
        self.pub_model_pose = rospy.Publisher('model_pose', Pose, latch=True, tcp_nodelay=True)

        self.trigarea_pub = rospy.Publisher('trigger_area', Polygon, latch=True, tcp_nodelay=True)

        self.pushover_pub = rospy.Publisher('note', String)
        self.save_pub = rospy.Publisher('save_object', UInt32)

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

        self.svg_pub = rospy.Publisher("svg_filename", String, latch=True)
        self.src_pub = rospy.Publisher("source", Vector3)
        self.ack_pub = rospy.Publisher("active", Bool)

		self.x0 = self.y0 = 0
        self.trigarea_pub.publish(
                nodelib.visualization.get_circle_trigger_volume_polygon(
                                        XFORM,
                                        START_RADIUS,self.x0,self.y0)
        )

        self.switch_conditions(None,force=START_CONDITION)

        self.timer = rospy.Timer(rospy.Duration(SWITCH_MODE_TIME),
                                  self.switch_conditions)

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

    def switch_conditions(self,event,force=None):
        if force is not None and type(force) is int:
            self.condition_n = force
        else:
            self.condition_n = (self.condition_n+1) % len(CONDITIONS)

        self.condition = CONDITIONS[self.condition_n]

        self.stimulus_filename = self.condition.split('/')[0]
        self.x0,self.y0 = map(float,self.condition.split('/')[1:-1])

        lag = float(self.condition.split('/')[-1])
        self.pub_lag.publish(lag)

        self.pub_model_pose.publish( self.get_model_pose_msg() )

        svg_filename = os.path.join(pkg_dir,"data","svgpaths",self.stimulus_filename[:-4])
        self.svg_pub.publish(svg_filename)

        self.log.stimulus_filename = self.stimulus_filename
        self.log.condition = self.condition
        self.log.startr = START_RADIUS
        rospy.loginfo('condition: %s (%f,%f)' % (self.condition,self.x0,self.y0))
        self.drop_lock_on()

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

                if (fly_z > Z_MAXIMUM) or (fly_z < Z_MINIMUM):
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

                # drop slow moving flies
                if now-self.last_check_flying_time > FLY_DIST_CHECK_TIME:
                    fly_dist = self.fly_dist
                    self.last_check_flying_time = now
                    self.fly_dist = 0
                    if fly_dist < FLY_DIST_MIN_DIST: # drop fly if it does not move enough
                        self.drop_lock_on()
                        rospy.loginfo('SLOW: too slow (%.3f < %.3f m/s)' % (fly_dist/FLY_DIST_CHECK_TIME, FLY_DIST_MIN_DIST/FLY_DIST_CHECK_TIME))
                        continue

                px,py = XFORM.xy_to_pxpy(fly_x,fly_y)
                self.src_pub.publish(px,py,fly_z)

            #don't need to record anything at the control rate
            #self.log.framenumber = framenumber
            #self.log.update()


            self.ack_pub.publish(active)

            r.sleep()

        rospy.loginfo('%s finished. saved data to %s' % (rospy.get_name(), self.log.close()))

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
        with self.trackinglock:
            rospy.loginfo('locked object %d at frame %d' % (obj.obj_id,framenumber))
            now = rospy.get_time()
            self.currently_locked_obj_id = obj.obj_id
            self.last_seen_time = now
            self.first_seen_time = now
            self.log.lock_object = obj.obj_id
            self.log.framenumber = framenumber
            self.last_check_flying_time = now

        self.pub_stimulus.publish( self.stimulus_filename )
        self.update()

    def drop_lock_on(self):
        with self.trackinglock:
            old_id = self.currently_locked_obj_id
            now = rospy.get_time()
            dt = now - self.first_seen_time

            rospy.loginfo('dropping locked object %s (tracked for %.1f)' % (old_id, dt))

            self.currently_locked_obj_id = None

            self.log.lock_object = IMPOSSIBLE_OBJ_ID
            self.log.framenumber = 0

        if (dt > 30) and (old_id is not None):
            if self.condition in COOL_CONDITIONS:
                self.pushover_pub.publish("Fly %s flew for %.1fs" % (old_id, dt))
                self.save_pub.publish(old_id)

        self.pub_stimulus.publish( HOLD_COND )
        self.update()        

def main():
    rospy.init_node("confinement")

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
