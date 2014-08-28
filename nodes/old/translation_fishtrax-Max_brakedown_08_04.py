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
from geometry_msgs.msg import Vector3, Pose
from ros_flydra.msg import flydra_mainbrain_super_packet

import flyflypath.model
import flyflypath.transform
import nodelib.log
import strawlab_freeflight_experiments.replay as sfe_replay

from strawlab_freeflight_experiments.topics import *

pkg_dir = roslib.packages.get_pkg_dir(PACKAGE)

CONTROL_RATE        = 80.0      #Hz
SWITCH_MODE_TIME    = 10*60    #alternate between control and static (i.e. experimental control) seconds

ADVANCE_RATIO       = 1/100.0   #### ????? ####

FLY_DIST_CHECK_TIME = 1       # time interval in seconds to check fly movement
FLY_DIST_MIN_DIST   = 0.001   # swimm more than 0.5 cm/sec # minimum distance fly must move in above interval to not be ignored

# start volume defined as cube
X_MIN = -0.15
X_MAX =  0.15
Y_MIN = -0.15
Y_MAX =  0.15
Z_MIN =  -0.10
Z_MAX =  0.03



# time interval in seconds to check fly movement after lock on
FLY_HEIGHT_CHECK_TIME = 1 #5 was good #0.5       

# z range for fly tracking (dropped outside)
# this range is only tested after the fly has been tracked for FLY_HEIGHT_CHECK_TIME seconds
Z_MINIMUM = -0.10 # 0.01 max
Z_MAXIMUM = 0.03 # 0.38 max

# y range for fly tracking (dropped outside)
# this range is only tested after the fly has been tracked for FLY_HEIGHT_CHECK_TIME seconds
Y_MINIMUM = -0.18
Y_MAXIMUM = 0.18



TIMEOUT             = 1
IMPOSSIBLE_OBJ_ID   = 0             #### ????? ####

PI = np.pi
TAU= 2*PI



#CONDITION =  svg_path(if omitted target = 0,0)/
#             gain/
#             advance_threshold(m)/
#             z_gain/
#             star_size
#             z_target
#
CONDITIONS = [
                "infinity03.svg/10.0/0.05/5/10.0/-0.01",  #best condition
                "infinity03.svg/10.0/0.05/5/10.0/-0.05",  #set z lower
                "infinity03.svg/10.0/0.05/5/5.0/-0.01",
                "infinity03.svg/10.0/0.05/5/20.0/-0.01",
                "infinity03.svg/20.0/0.05/5/10.0/-0.01",
                "infinity03.svg/5.0/0.05/5/10.0/-0.01"
]

START_CONDITION = CONDITIONS[0]
#If there is a considerable flight in these conditions then a pushover
#message is sent and a video recorded
COOL_CONDITIONS = set(CONDITIONS[0:3])
MAX_COOL = 10

XFORM = flyflypath.transform.SVGTransform()

class Logger(nodelib.log.CsvLogger):
    STATE = ("trg_x","trg_y","trg_z","cyl_x","cyl_y","cyl_r","ratio","stim_x","stim_y","stim_z")

class Node(object):
    def __init__(self, wait_for_flydra, use_tmpdir, continue_existing):

        self._pub_stim_mode = display_client.DisplayServerProxy.set_stimulus_mode(
            'StimulusStarField')

        self.pub_velocity = rospy.Publisher(TOPIC_STAR_VELOCITY, Vector3, latch=True, tcp_nodelay=True)
        self.pub_size = rospy.Publisher(TOPIC_STAR_SIZE, Float32, latch=True, tcp_nodelay=True)

        self.pub_pushover = rospy.Publisher('note', String)
        self.pub_save = rospy.Publisher('save_object', UInt32)

        self.pub_velocity.publish(0,0,0)
        self.pub_size.publish(5.0)

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

        self.n_cool = 0

        #start criteria for experiment
        self.x0 = self.y0 = 0
        #target (for moving points)
        self.trg_x = self.trg_y = 0.0

        self.svg_pub = rospy.Publisher("svg_filename", String, latch=True)
        self.src_pub = rospy.Publisher("source", Vector3)
        self.trg_pub = rospy.Publisher("target", Vector3)
        self.ack_pub = rospy.Publisher("active", Bool)

        self.switch_conditions(None,force=START_CONDITION)

        self.timer = rospy.Timer(rospy.Duration(SWITCH_MODE_TIME),
                                  self.switch_conditions)

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

        svg,p,advance,v_gain,star_size,z_target = self.condition.split('/')
        self.p_const = float(p)
        self.v_gain = float(v_gain)
        self.advance_px = XFORM.m_to_pixel(float(advance))
        self.z_target = float(z_target)    #!!!z = 0 is surface

        if str(svg):
            self.svg_fn = os.path.join(pkg_dir,'data','svgpaths', str(svg))
            self.model = flyflypath.model.MovingPointSvgPath(self.svg_fn)
            self.svg_pub.publish(self.svg_fn)
        else:
            self.svg_fn = ''

        self.pub_size.publish(float(star_size))

        rospy.loginfo('condition: %s (p=%.1f, svg=%s, advance=%.1fpx)' % (self.condition,self.p_const,os.path.basename(self.svg_fn),self.advance_px))

    def get_v_rate(self,fly_z):
        #return early if this is a replay experiment
        if self.is_replay_experiment_z:
            return self.replay_z.next()

        return self.v_gain*(self.z_target - fly_z)

    def get_starfield_velocity_vector(self,fly_x,fly_y,fly_z, fly_vx, fly_vy, fly_vz):
        if self.svg_fn and (not self.is_replay_experiment_rotation):
            with self.trackinglock:
                px,py = XFORM.xy_to_pxpy(fly_x,fly_y)
                segment = self.model.connect_to_moving_point(p=None, px=px,py=py)
                if segment.length < self.advance_px:
                    self.log.ratio, newpt = self.model.advance_point(ADVANCE_RATIO, wrap=True)
                    self.trg_x,self.trg_y = XFORM.pxpy_to_xy(newpt.x,newpt.y)
                    self.ratio_total += ADVANCE_RATIO
        else:
            self.trg_x = self.trg_y = 0.0

        #return early if this is a replay experiment
        if self.is_replay_experiment_rotation:
            return self.replay_rotation.next(), self.trg_x,self.trg_y

        dx = self.trg_x-fly_x
        dy = self.trg_y-fly_y

        return self.p_const*dx,self.p_const*dy,self.trg_x,self.trg_y

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

                rate_x,rate_y,trg_x,trg_y = self.get_starfield_velocity_vector(fly_x, fly_y, fly_z, fly_vx, fly_vy, fly_vz)
                v_rate = self.get_v_rate(fly_z)

                px,py = XFORM.xy_to_pxpy(fly_x,fly_y)
                self.src_pub.publish(px,py,fly_z)
                trg_px, trg_py = XFORM.xy_to_pxpy(trg_x,trg_y)
                self.trg_pub.publish(trg_px,trg_py,self.z_target)

                self.log.trg_x = trg_x; self.log.trg_y = trg_y; self.log.trg_z = self.z_target

                self.log.stim_x = rate_x
                self.log.stim_y = rate_y
                self.log.stim_z = v_rate
                self.pub_velocity.publish(rate_x,rate_y,v_rate)

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

            if self.svg_fn:
                px,py = XFORM.xy_to_pxpy(obj.position.x,obj.position.y)
                closest,ratio = self.model.connect_closest(p=None, px=px, py=py)
                self.log.ratio,newpt = self.model.start_move_from_ratio(ratio)
                self.trg_x,self.trg_y = XFORM.pxpy_to_xy(newpt.x,newpt.y)
            else:
                self.log.ratio = 0
                self.trg_x = self.trg_y = 0.0

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

        self.pub_velocity.publish(0,0,0)

        if (self.ratio_total > 2) and (old_id is not None):
            if self.condition in COOL_CONDITIONS:
                if self.n_cool < MAX_COOL:
                    self.pub_pushover.publish("Fly %s flew %.1f loops (in %.1fs)" % (old_id, self.ratio_total, dt))
                    self.pub_save.publish(old_id)
                    self.n_cool += 1

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

