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
from geometry_msgs.msg import Vector3
from ros_flydra.msg import flydra_mainbrain_super_packet

import flyflypath.model
import flyflypath.transform
import nodelib.node
import strawlab_freeflight_experiments.replay as sfe_replay
import strawlab_freeflight_experiments.perturb as sfe_perturb
import strawlab_freeflight_experiments.conditions as sfe_conditions

from strawlab_freeflight_experiments.topics import *

pkg_dir = roslib.packages.get_pkg_dir(PACKAGE)

CONTROL_RATE        = 80.0      #Hz

ADVANCE_RATIO       = 1/100.0

FLY_DIST_CHECK_TIME = 0.2       # time interval in seconds to check fly movement
FLY_DIST_MIN_DIST   = 0.0005    # minimum distance fly must move in above interval to not be ignored

# start volume defined as cube
X_MIN = -0.30
X_MAX =  0.30
Y_MIN = -0.15
Y_MAX =  0.15
Z_MIN =  0.01
Z_MAX =  0.38

# time interval in seconds to check fly movement after lock on
FLY_HEIGHT_CHECK_TIME = 0.5       

# z range for fly tracking (dropped outside)
# this range is only tested after the fly has been tracked for FLY_HEIGHT_CHECK_TIME seconds
Z_MINIMUM = 0.01
Z_MAXIMUM = 0.38

# y range for fly tracking (dropped outside)
# this range is only tested after the fly has been tracked for FLY_HEIGHT_CHECK_TIME seconds
Y_MINIMUM = -0.18
Y_MAXIMUM = 0.18


GRAY_FN = "gray.png"

TIMEOUT             = 0.5
IMPOSSIBLE_OBJ_ID   = 0

PI = np.pi
TAU= 2*PI

MAX_ROTATION_RATE = 3


XFORM = flyflypath.transform.SVGTransform()

class Node(nodelib.node.Experiment):
    def __init__(self, args):
        super(Node, self).__init__(args=args,
                                   state=("rotation_rate","trg_x","trg_y","trg_z","cyl_x","cyl_y","cyl_r","ratio","v_offset_rate","perturb_progress"))

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

        self.pub_rotation.publish(0)
        self.pub_v_offset_value.publish(0)

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
            self.model = None

            self.ratio_total = 0

            self.replay_rotation = sfe_replay.ReplayStimulus(default=0.0)
            self.replay_z = sfe_replay.ReplayStimulus(default=0.0)

            self.blacklist = {}

        #start criteria for experiment
        self.x0 = self.y0 = 0
        #target (for moving points)
        self.trg_x = self.trg_y = 0.0

        self.svg_pub = rospy.Publisher("svg_filename", String, latch=True)
        self.src_pub = rospy.Publisher("source", Vector3)
        self.trg_pub = rospy.Publisher("target", Vector3)
        self.ack_pub = rospy.Publisher("active", Bool)

        self.switch_conditions()

        rospy.Subscriber("flydra_mainbrain/super_packets",
                         flydra_mainbrain_super_packet,
                         self.on_flydra_mainbrain_super_packets)

    def is_perturbation_experiment(self, what):
        return (not isinstance(self.perturber, sfe_perturb.NoPerturb)) and self.perturber.what == what

    def is_replay_experiment(self, what):
        if what == 'rotation_rate':
            return np.isnan(self.p_const)
        elif what == 'z':
            return np.isnan(self.v_gain)
        else:
            raise Exception("Unsupported replay configuration")

    def switch_conditions(self):

        self.drop_lock_on()

        ssvg            = str(self.condition['svg_path'])
        self.img_fn     = str(self.condition['cylinder_image'])
        self.p_const    = float(self.condition['gain'])
        self.v_gain     = float(self.condition['z_gain'])
        self.rad_locked = float(self.condition['radius_when_locked'])
        self.advance_px = XFORM.m_to_pixel(float(self.condition['advance_threshold']))
        self.z_target   = float(self.condition['z_target'])
        if self.condition.is_type('perturbation'):
            perturb_desc = str(self.condition['perturb_desc'])
        else:
            perturb_desc = None

        self.log.cyl_r = self.rad_locked

        #default to no perturb
        self.perturber = sfe_perturb.NoPerturb()

        self.svg_fn = ''
        if ssvg:
            self.svg_fn = os.path.join(pkg_dir,'data','svgpaths', ssvg)
            self.model = flyflypath.model.MovingPointSvgPath(self.svg_fn)
            self.svg_pub.publish(self.svg_fn)

            self.perturber = sfe_perturb.get_perturb_object(perturb_desc)

        self.rotation_rate_max = float(self.condition.get('rotation_rate_max', MAX_ROTATION_RATE))

        #HACK
        self.pub_cyl_height.publish(np.abs(5*self.rad_locked))
        
        rospy.loginfo('condition: %s (p=%.1f, svg=%s, rad locked=%.1f advance=%.1fpx)' % (self.condition.name,self.p_const,os.path.basename(self.svg_fn),self.rad_locked,self.advance_px))
        rospy.loginfo('perturbation: %r' % self.perturber)

    def get_v_rate(self, fly_x, fly_y, fly_z, fly_vx, fly_vy, fly_vz, now, framenumber, currently_locked_obj_id):
        #return early if open loop
        if self.is_perturbation_experiment('z'):
            if self.perturber.should_perturb(fly_x, fly_y, fly_z, fly_vx, fly_vy, fly_vz,
                                             self.model.ratio, self.ratio_total,
                                             now, now - self.first_seen_time,
                                             framenumber, currently_locked_obj_id):
                rate,state = self.perturber.step(
                                             fly_x, fly_y, fly_z, fly_vx, fly_vy, fly_vz,
                                             now, now - self.first_seen_time,
                                             framenumber, currently_locked_obj_id)

                if state=='finished':
                    self.drop_lock_on(blacklist=True)
                    rospy.loginfo("'z' perturbation finished")

                return rate

        #return early if this is a replay experiment
        if self.is_replay_experiment('z'):
            return self.replay_z.next()

        return self.v_gain*(fly_z-self.z_target)

    def get_rotation_velocity_vector(self, fly_x, fly_y, fly_z, fly_vx, fly_vy, fly_vz, now, framenumber, currently_locked_obj_id):
        could_perturb = False
        if self.svg_fn and (not self.is_replay_experiment('rotation_rate')):
            could_perturb = self.is_perturbation_experiment('rotation_rate')
            with self.trackinglock:
                px,py = XFORM.xy_to_pxpy(fly_x,fly_y)
                segment = self.model.connect_to_moving_point(p=None, px=px,py=py)
                if segment.length < self.advance_px:
                    new_ratio, newpt = self.model.advance_point(ADVANCE_RATIO, wrap=True)
                    self.trg_x,self.trg_y = XFORM.pxpy_to_xy(newpt.x,newpt.y)
                    self.ratio_total += ADVANCE_RATIO
                    self.log.ratio = new_ratio
        else:
            self.trg_x = self.trg_y = 0.0

        #return early if open loop
        if could_perturb:
            if self.perturber.should_perturb(fly_x, fly_y, fly_z, fly_vx, fly_vy, fly_vz,
                                             self.model.ratio, self.ratio_total,
                                             now, now - self.first_seen_time,
                                             framenumber, currently_locked_obj_id):
                rate,state = self.perturber.step(
                                             fly_x, fly_y, fly_z, fly_vx, fly_vy, fly_vz,
                                             now, now - self.first_seen_time,
                                             framenumber, currently_locked_obj_id)

                if state=='finished':
                    self.drop_lock_on(blacklist=True)
                    if abs(fly_z-self.z_target) < 0.1:
                        self.save_cool_condition(currently_locked_obj_id, note="Fly %s completed perturbation" % currently_locked_obj_id)
                    rospy.loginfo('perturbation finished')

                return rate, self.trg_x,self.trg_y

        #return early if this is a replay experiment
        if self.is_replay_experiment('rotation_rate'):
            return self.replay_rotation.next(), self.trg_x,self.trg_y

        dpos = np.array((self.trg_x-fly_x,self.trg_y-fly_y))
        vel  = np.array((fly_vx, fly_vy))

        speed = np.linalg.norm(vel)

        dposn = dpos / np.linalg.norm(dpos)
        eps = 1e-20
        if speed > eps:
            veln = vel / speed

            vel_angle = np.arctan2( veln[1], veln[0] )
            desired_angle = np.arctan2( dposn[1], dposn[0] )

            error_angle_unwrapped = desired_angle - vel_angle

            error_angle = (error_angle_unwrapped + PI) % TAU - PI

            velocity_gate = max( speed*20.0, 1.0) # linear ramp to 1.0
            val = velocity_gate*error_angle*self.p_const

            #print '--------'
            #print 'vel_angle',vel_angle
            #print 'desired_angle', desired_angle
            #print 'error_angle',error_angle
            #print
        else:
            val = 0.0

        val = np.clip(val,-self.rotation_rate_max,self.rotation_rate_max)

        return val,self.trg_x,self.trg_y

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

                rate,trg_x,trg_y = self.get_rotation_velocity_vector(fly_x, fly_y, fly_z, fly_vx, fly_vy, fly_vz, now, framenumber, currently_locked_obj_id)
                v_rate = self.get_v_rate(fly_x, fly_y, fly_z, fly_vx, fly_vy, fly_vz, now, framenumber, currently_locked_obj_id)

                px,py = XFORM.xy_to_pxpy(fly_x,fly_y)
                self.src_pub.publish(px,py,fly_z)
                trg_px, trg_py = XFORM.xy_to_pxpy(trg_x,trg_y)
                self.trg_pub.publish(trg_px,trg_py,self.z_target)

                self.log.cyl_x = fly_x; self.log.cyl_y = fly_y;
                self.log.trg_x = trg_x; self.log.trg_y = trg_y; self.log.trg_z = self.z_target

                self.log.perturb_progress = self.perturber.progress

                self.log.rotation_rate = rate
                self.pub_rotation_velocity.publish(rate)

                self.log.v_offset_rate = v_rate
                self.pub_v_offset_rate.publish(v_rate)

                if self.rad_locked > 0:
                    self.pub_cyl_centre.publish(fly_x,fly_y,0)
                else:
                    self.pub_cyl_centre.publish(0,0,0)

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

            self.perturber.reset()
            self.replay_rotation.reset()
            self.replay_z.reset()

        self.pub_image.publish(self.img_fn)
        self.pub_cyl_radius.publish(np.abs(self.rad_locked))

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

            self.log.rotation_rate = 0
            self.log.ratio = 0

            self.log.cyl_r = 0.5

            self.log.cyl_x = 0
            self.log.cyl_y = 0

            if blacklist:
                self.blacklist[old_id] = True

        self.pub_image.publish(GRAY_FN)
        self.pub_rotation_velocity.publish(0)
        self.pub_cyl_radius.publish(0.5)
        self.pub_cyl_centre.publish(0,0,0)

        if (self.ratio_total > 2) and (old_id is not None):
            self.save_cool_condition(old_id, note="Fly %s flew %.1f loops (in %.1fs)" % (old_id, self.ratio_total, dt))

        self.update()

def main():
    rospy.init_node("perturbation")
    parser, args = nodelib.node.get_and_parse_commandline()
    node = Node(args)
    return node.run()

if __name__=='__main__':
    main()

