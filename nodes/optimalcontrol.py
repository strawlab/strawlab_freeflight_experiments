#!/usr/bin/env python

import sys
import math
import numpy as np
import threading
import argparse
import os.path

from profilehooks import timecall
from rawtime import monotonic_time

import roslib
import roslib.packages
roslib.load_manifest('strawlab_freeflight_experiments')
pkg_dir = roslib.packages.get_pkg_dir('strawlab_freeflight_experiments')

import rospy
import flyvr.display_client as display_client
from std_msgs.msg import UInt32, Bool, Float32, String
from geometry_msgs.msg import Vector3, Pose, Polygon, Point32

from ros_flydra.msg import flydra_mainbrain_super_packet
from ros_flydra.constants import IMPOSSIBLE_OBJ_ID

import flyflypath.model
import nodelib.log

from strawlab_freeflight_experiments.topics import *
from strawlab_freeflight_experiments.controllers import MPC
from strawlab_freeflight_experiments.controllers.util import Fly

CONTROL_RATE        = 80.0      #Hz
SWITCH_MODE_TIME    = 5.0*60    #alternate between control and static (i.e. experimental control) seconds

FLY_DIST_CHECK_TIME = 5.0
FLY_DIST_MIN_DIST   = 0.2

START_RADIUS    = 0.35
START_ZDIST     = 0.4
START_Z         = 0.5

# z range for fly tracking (dropped outside)
Z_MINIMUM = 0.00
Z_MAXIMUM = 0.95

GRAY_FN = "gray.png"

TIMEOUT             = 0.5

PI = np.pi
TAU= 2*PI

MAX_ROTATION_RATE = 1.5

CONTROL_IN_PARALLEL = True

#CONDITION = "cylinder_image/
#             svg_path(if omitted target = 0,0)/
#             gain/
#             radius_when_locked(+ve = centre of cylinder is locked to the fly)/
#             advance_threshold(m)/
#             z_gain"
#
CONDITIONS = [
              "checkerboard16.png//nan/-10.0/nan/0.20",
              "gray.png//nan/-10.0/nan/0.20",
]
START_CONDITION = CONDITIONS[0]
#If there is a considerable flight in these conditions then a pushover
#message is sent and a video recorded
COOL_CONDITIONS = set()

class Logger(nodelib.log.CsvLogger):
    STATE = ("rotation_rate","trg_x","trg_y","trg_z","cyl_x","cyl_y","cyl_r","ratio","v_offset_rate","j","w","path_theta","ekf_en","control_en","t2_5ms","xest0","xest1","xest2","xest3","xest4")

class Node(object):
    def __init__(self, wait_for_flydra, use_tmpdir, continue_existing):

        self._pub_stim_mode = display_client.DisplayServerProxy.set_stimulus_mode(
            'StimulusCylinder')

        self.pub_rotation = rospy.Publisher(TOPIC_CYL_ROTATION, Float32, latch=True, tcp_nodelay=True)
        self.pub_rotation_velocity = rospy.Publisher(TOPIC_CYL_ROTATION_RATE, Float32, latch=True, tcp_nodelay=True)
        self.pub_v_offset_value = rospy.Publisher(TOPIC_CYL_V_OFFSET_VALUE, Float32, latch=True, tcp_nodelay=True)
        self.pub_v_offset_rate = rospy.Publisher(TOPIC_CYL_V_OFFSET_RATE, Float32, latch=True, tcp_nodelay=True)
        self.pub_image = rospy.Publisher(TOPIC_CYL_IMAGE, String, latch=True, tcp_nodelay=True)
        self.pub_cyl_centre = rospy.Publisher(TOPIC_CYL_CENTRE, Vector3, latch=True, tcp_nodelay=True)
        self.pub_cyl_radius = rospy.Publisher(TOPIC_CYL_RADIUS, Float32, latch=True, tcp_nodelay=True)
        self.pub_cyl_height = rospy.Publisher(TOPIC_CYL_HEIGHT, Float32, latch=True, tcp_nodelay=True)

        #publish for the follow_path monitor
        self.svg_pub = rospy.Publisher("svg_filename", String, latch=True)
        self.src_m_pub = rospy.Publisher("source_m", Vector3)
        self.trg_m_pub = rospy.Publisher("target_m", Vector3)
        self.path_m_pub = rospy.Publisher('path_m', Polygon, latch=True, tcp_nodelay=True)

        self.pub_pushover = rospy.Publisher('note', String)
        self.pub_save = rospy.Publisher('save_object', UInt32)

        self.pub_rotation.publish(0)
        self.pub_v_offset_value.publish(0)

        self.pub_lock_object = rospy.Publisher('lock_object', UInt32, latch=True, tcp_nodelay=True)
        self.pub_lock_object.publish(IMPOSSIBLE_OBJ_ID)

        self.log = Logger(wait=wait_for_flydra, use_tmpdir=use_tmpdir, continue_existing=continue_existing)
        self.log.ratio = 0 #backwards compatibility - see path_theta for mpc path parameter

        #setup the MPC controller
        self.controllock = threading.Lock()
        with self.controllock:
            self.control = MPC.MPC()
            self.control.reset()
            #rotation_rate = omega

        #publish the path
        self.path_m_pub.publish(Polygon(points=[Point32(x,y,0) for x,y in self.control.path]))

        #protect the tracked id and fly position between the time syncronous main loop and the asyn
        #tracking/lockon/off updates
        self.trackinglock = threading.Lock()
        with self.trackinglock:
            self.currently_locked_obj_id = IMPOSSIBLE_OBJ_ID
            self.fly = Vector3()
            self.flyv = Vector3()
            now = rospy.get_time()
            self.first_seen_time = now
            self.last_seen_time = now
            self.last_fly_x = self.fly.x; self.last_fly_y = self.fly.y; self.last_fly_z = self.fly.z;
            self.last_check_flying_time = now
            self.fly_dist = 0

            self.ratio_total = 0

        #start criteria for experiment
        self.x0 = self.y0 = 0

        self.ack_pub = rospy.Publisher("active", Bool)

        self.switch_conditions(None,force=START_CONDITION)

        self.timer = rospy.Timer(rospy.Duration(SWITCH_MODE_TIME),
                                  self.switch_conditions)

        rospy.Subscriber("flydra_mainbrain/super_packets",
                         flydra_mainbrain_super_packet,
                         self.on_flydra_mainbrain_super_packets)

    @timecall(stats=1000, immediate=False, timer=monotonic_time)
    def do_control(self):
        if CONTROL_IN_PARALLEL:
            self.control.run_control()
        else:
            with self.controllock:
                self.control.run_control()
        self.log.j = self.control._CT_jout.value
        self.log.w = self.control._CT_wout.value
        self.log.path_theta = self.control._CT_thetaout.value
        self.log.trg_x, self.log.trg_y = self.control.target_point

    @timecall(stats=1000, immediate=False, timer=monotonic_time)
    def do_update_ekf(self, x,y):
        self.control.run_ekf(None,x,y)
        #FIXME: log x here for martin

    @timecall(stats=1000, immediate=False, timer=monotonic_time)
    def do_calculate_input(self):
        if CONTROL_IN_PARALLEL:
            self.control.run_calculate_input()
        else:
            with self.controllock:
                self.control.run_calculate_input()

    def switch_conditions(self,event,force=''):
        if force:
            self.condition = force
        else:
            i = CONDITIONS.index(self.condition)
            j = (i+1) % len(CONDITIONS)
            self.condition = CONDITIONS[j]
        self.log.condition = self.condition

        self.drop_lock_on()

        img,svg,p,rad,advance,v_gain = self.condition.split('/')
        self.img_fn = str(img)
        self.p_const = float(p)
        self.v_gain = float(v_gain)
        self.rad_locked = float(rad)
        self.z_target = 0.7
        self.svg_fn = str(svg)
        self.svg_pub.publish(self.svg_fn)

        self.log.trg_z = self.z_target
        self.log.cyl_r = self.rad_locked

        #HACK
        self.pub_cyl_height.publish(np.abs(5*self.rad_locked))
        
        rospy.loginfo('condition: %s (p=%.1f, svg=%s, rad locked=%.1f)' % (self.condition,self.p_const,os.path.basename(self.svg_fn),self.rad_locked))

    def get_v_rate(self,fly_z):
        return self.v_gain*(fly_z-self.z_target)

    def run(self):
        rospy.loginfo('running stimulus')
        #stick close to matlab/simulink, so run this loop at the fastest
        #rate (2.5ms) and do our own scheduling
        r = rospy.Rate(1000/2.5)
        i = 0
        t0 = monotonic_time()

        cthread = None

        while not rospy.is_shutdown():
            with self.trackinglock:
                currently_locked_obj_id = self.currently_locked_obj_id
                fly_x = self.fly.x
                fly_y = self.fly.y
                fly_z = self.fly.z
                fly_v = math.sqrt( (self.flyv.x ** 2) + (self.flyv.y ** 2) )

            #lost fly position
            if np.isnan(fly_z):
                continue

            if currently_locked_obj_id != IMPOSSIBLE_OBJ_ID:
                i += 1

                self.log.ekf_en = self.control.ekf_enabled
                self.log.control_en = self.control.controller_enabled

                #5ms
                if (i % 2) == 0:
                    if self.control.ekf_enabled:
                        self.do_update_ekf(fly_x, fly_y)

                        xest = self.control._ekf_xest.flatten()
                        for _i in range(5):
                            setattr(self.log, "xest%d"%_i, xest[_i])

                #12.5ms
                if (i % 5) == 0:
                    self.do_calculate_input()

                    #lets do the vertical control here too,
                    #at the same rate, 80Hz, as old
                    v_rate = self.get_v_rate(fly_z)
                    self.log.v_offset_rate = v_rate
                    self.pub_v_offset_rate.publish(v_rate)

                    rotation_rate = self.control.rotation_rate
                    self.log.rotation_rate = rotation_rate

                    trg_x, trg_y = self.control.target_point
                    self.trg_m_pub.publish(trg_x,trg_y,self.z_target)

                    #print rotation_rate, fly_v, fly_z

                    if np.isnan(rotation_rate):
                        with self.controllock:
                            self.control.reset()
                    else:
                        self.pub_rotation_velocity.publish(rotation_rate)

                    self.src_m_pub.publish(fly_x,fly_y,fly_z)
                    self.ack_pub.publish(self.control.controller_enabled > 0)

                #50ms
                if (i % 20) == 0:
                    if cthread is not None:
                        cthread.join()

                    cthread = threading.Thread(target=self.do_control)
                    cthread.start()

                self.log.t2_5ms = i
                self.log.update()

            r.sleep()

        rospy.loginfo('%s finished. saved data to %s' % (rospy.get_name(), self.log.close()))

    def is_in_trigger_volume(self, obj):
        c = np.array( (self.x0,self.y0) )
        p = np.array( (obj.position.x, obj.position.y) )
        dist = np.sqrt(np.sum((c-p)**2))
        if (dist < START_RADIUS) and (abs(obj.position.z-START_Z) < START_ZDIST):
            return True
        return False

    def maybe_drop_lock_on(self, now, fly_x, fly_y, fly_z):
        dt = now - self.last_seen_time
        if dt > TIMEOUT:
            rospy.loginfo('TIMEOUT: time since last seen > %.1fs (%.3fs)' % (TIMEOUT,dt))
            return True

        if (fly_z > Z_MAXIMUM) or (fly_z < Z_MINIMUM):
            return True

        if np.isnan(fly_x):
            #we have a race  - a fly to track with no pose yet
            return False

        #distance accounting, give up on fly if it is not moving
        self.fly_dist += math.sqrt((fly_x-self.last_fly_x)**2 +
                                   (fly_y-self.last_fly_y)**2 +
                                   (fly_z-self.last_fly_z)**2)
        self.last_fly_x = fly_x; self.last_fly_y = fly_y; self.last_fly_z = fly_z;

        # drop slow moving flies
        if (now - self.last_check_flying_time) > FLY_DIST_CHECK_TIME:
            fly_dist = self.fly_dist
            self.last_check_flying_time = now
            self.fly_dist = 0
            if fly_dist < FLY_DIST_MIN_DIST: # drop fly if it does not move enough
                rospy.loginfo('SLOW: too slow (%.3f < %.3f m/s)' % (fly_dist/FLY_DIST_CHECK_TIME, FLY_DIST_MIN_DIST/FLY_DIST_CHECK_TIME))
                return True

        return False

    def on_flydra_mainbrain_super_packets(self,data):
        now = rospy.get_time()
        flies = []

        with self.trackinglock:
            while True:
                for packet in data.packets:
                    for obj in packet.objects:
                        if self.currently_locked_obj_id == IMPOSSIBLE_OBJ_ID:
                            if self.is_in_trigger_volume(obj):
                                flies.append( Fly.from_flydra_object(obj) )
                        elif obj.obj_id == self.currently_locked_obj_id:
                            self.last_seen_time = now
                            self.fly = obj.position
                            self.flyv = obj.velocity
                            self.log.framenumber = packet.framenumber
                            break
                break

            currently_locked_obj_id = self.currently_locked_obj_id

        with self.controllock:
            if currently_locked_obj_id != IMPOSSIBLE_OBJ_ID:
                if self.maybe_drop_lock_on(now, self.fly.x, self.fly.y, self.fly.z):
                    self.drop_lock_on()
                    self.control.reset()
            elif flies:
                obj_id = self.control.should_control(flies)
                if obj_id != -1:
                    self.lock_on(obj_id, packet.framenumber)

    def update_lock_on_off(self):
        self.log.update()
        self.pub_lock_object.publish(self.currently_locked_obj_id)

    def lock_on(self,obj_id,framenumber):
        rospy.loginfo('locked object %d at frame %d' % (obj_id,framenumber))
        now = rospy.get_time()
        self.currently_locked_obj_id = obj_id
        self.last_seen_time = now
        self.first_seen_time = now
        self.last_check_flying_time = now

        self.log.lock_object = obj_id
        self.log.framenumber = framenumber

        self.pub_image.publish(self.img_fn)
        self.pub_cyl_radius.publish(np.abs(self.rad_locked))

        self.update_lock_on_off()

    def drop_lock_on(self):
        old_id = self.currently_locked_obj_id
        now = rospy.get_time()
        dt = now - self.first_seen_time

        rospy.loginfo('dropping locked object %s (tracked for %.1f)' % (old_id, dt))

        self.currently_locked_obj_id = IMPOSSIBLE_OBJ_ID

        self.log.lock_object = IMPOSSIBLE_OBJ_ID
        self.log.framenumber = 0
        self.log.rotation_rate = 0
        self.log.ratio = 0
        self.log.cyl_r = 0.5
        self.log.cyl_x = 0
        self.log.cyl_y = 0

        self.pub_image.publish(GRAY_FN)
        self.pub_rotation_velocity.publish(0)
        self.pub_cyl_radius.publish(0.5)
        self.pub_cyl_centre.publish(0,0,0)
        self.ack_pub.publish(False)

        self.update_lock_on_off()

def main():
    rospy.init_node("rotation")

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

