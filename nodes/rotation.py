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
import rospkg

import flyflypath.model
import flyflypath.transform
import nodelib.log

pkg_dir = roslib.packages.get_pkg_dir(PACKAGE)

TOPIC_CYL_ROTATION      = "cylinder_rotation"
TOPIC_CYL_ROTATION_RATE = "cylinder_rotation_rate"
TOPIC_CYL_V_OFFSET_VALUE= "cylinder_v_offset_value"
TOPIC_CYL_V_OFFSET_RATE = "cylinder_v_offset_rate"
TOPIC_CYL_IMAGE         = "cylinder_image"
TOPIC_CYL_CENTRE        = "cylinder_centre"
TOPIC_CYL_RADIUS        = "cylinder_radius"
TOPIC_CYL_HEIGHT        = "cylinder_height"

CONTROL_RATE        = 80.0      #Hz
SWITCH_MODE_TIME    = 5.0*60    #alternate between control and static (i.e. experimental control) seconds

ADVANCE_RATIO       = 1/100.0

FLY_DIST_CHECK_TIME = 5.0
FLY_DIST_MIN_DIST   = 0.2

START_RADIUS    = 0.35
START_ZDIST     = 0.4
START_Z         = 0.5

GRAY_FN = "gray.png"

TIMEOUT             = 0.5
IMPOSSIBLE_OBJ_ID   = 0

PI = np.pi
TAU= 2*PI

MAX_ROTATION_RATE = 1.5

#CONDITION = "cylinder_image/
#             svg_path(if omitted target = 0,0)/
#             gain/
#             radius_when_locked(+ve = centre of cylinder is locked to the fly)/
#             advance_threshold(m)/
#             v_gain"
#
CONDITIONS = [
#              "checkerboard16.png/infinity.svg/+0.3/+0.2/0.1/0.20",
              "checkerboard16.png/infinity.svg/+0.3/-10.0/0.1/0.20",
#              "checkerboard16.png/infinity.svg/+0.3/-5.0/0.1/0.20",
#              "checkerboard16.png/infinity.svg/+0.3/-2.0/0.1/0.20",
#              "checkerboard16.png/infinity.svg/+0.3/-1.0/0.1/0.20",
#              "checkerboard16.png/infinity.svg/+0.3/-0.5/0.1/0.20",
              "checkerboard16.png/infinity.svg/+0.0/-10.0/0.1/0.00",
              "gray.png/infinity.svg/+0.3/-10.0/0.1/0.20",
]
START_CONDITION = CONDITIONS[0]
#If there is a considerable flight in these conditions then a pushover
#message is sent and a video recorded
COOL_CONDITIONS = set()#set(CONDITIONS[0:])

XFORM = flyflypath.transform.SVGTransform()

class Logger(nodelib.log.CsvLogger):
    STATE = ("rotation_rate","trg_x","trg_y","trg_z","cyl_x","cyl_y","cyl_r","ratio","v_offset_rate")

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

        self.pub_pushover = rospy.Publisher('note', String)
        self.pub_save = rospy.Publisher('save_object', UInt32)

        self.pub_rotation.publish(0)
        self.pub_v_offset_value.publish(0)

        self.pub_lock_object = rospy.Publisher('lock_object', UInt32, latch=True, tcp_nodelay=True)
        self.pub_lock_object.publish(IMPOSSIBLE_OBJ_ID)

        self.log = Logger(wait=wait_for_flydra, use_tmpdir=use_tmpdir, continue_existing=continue_existing)

        #protect the traked id and fly position between the time syncronous main loop and the asyn
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
        self.advance_px = XFORM.m_to_pixel(float(advance))
        
        self.log.cyl_r = self.rad_locked

        if str(svg):
            self.svg_fn = os.path.join(pkg_dir,'data','svgpaths', str(svg))
            self.model = flyflypath.model.MovingPointSvgPath(self.svg_fn)
            self.svg_pub.publish(self.svg_fn)
        else:
            self.svg_fn = ''

        if self.rad_locked < 0:
            #HACK
            self.pub_cyl_height.publish(-4.5*self.rad_locked)
        else:
            self.pub_cyl_height.publish(1.0)
        
        rospy.loginfo('condition: %s (p=%.1f, svg=%s, rad locked=%.1f advance=%.1fpx)' % (self.condition,self.p_const,os.path.basename(self.svg_fn),self.rad_locked,self.advance_px))

    def get_v_rate(self,fly_z):
        target_z = 0.7
        return self.v_gain*(fly_z-target_z)

    def get_rotation_velocity_vector(self,fly_x,fly_y,fly_z, fly_vx, fly_vy, fly_vz):
        if self.svg_fn:
            with self.trackinglock:
                px,py = XFORM.xy_to_pxpy(fly_x,fly_y)
                segment = self.model.connect_to_moving_point(p=None, px=px,py=py)
                if segment.length < self.advance_px:
                    self.log.ratio, newpt = self.model.advance_point(ADVANCE_RATIO, wrap=True)
                    self.trg_x,self.trg_y = XFORM.pxpy_to_xy(newpt.x,newpt.y)
                    self.ratio_total += ADVANCE_RATIO
        else:
            self.trg_x = self.trg_y = 0.0

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

        val = np.clip(val,-MAX_ROTATION_RATE,MAX_ROTATION_RATE)

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

                rate,trg_x,trg_y = self.get_rotation_velocity_vector(fly_x, fly_y, fly_z, fly_vx, fly_vy, fly_vz)
                v_rate = self.get_v_rate(fly_z)

                px,py = XFORM.xy_to_pxpy(fly_x,fly_y)
                self.src_pub.publish(px,py,fly_z)
                trg_px, trg_py = XFORM.xy_to_pxpy(trg_x,trg_y)
                self.trg_pub.publish(trg_px,trg_py,START_Z)

                self.log.cyl_x = fly_x; self.log.cyl_y = fly_y;
                self.log.trg_x = trg_x; self.log.trg_y = trg_y; self.log.trg_z = START_Z

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
                        self.flyv = obj.velocity
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

            if self.svg_fn:
                px,py = XFORM.xy_to_pxpy(obj.position.x,obj.position.y)
                closest,ratio = self.model.connect_closest(p=None, px=px, py=py)
                self.log.ratio,newpt = self.model.start_move_from_ratio(ratio)
                self.trg_x,self.trg_y = XFORM.pxpy_to_xy(newpt.x,newpt.y)
            else:
                self.log.ratio = 0
                self.trg_x = self.trg_y = 0.0

            self.ratio_total = 0

        self.pub_image.publish( self.img_fn )

        self.pub_cyl_radius.publish(np.abs(self.rad_locked))

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

            self.log.rotation_rate = 0
            self.log.ratio = 0

            self.log.cyl_r = 0.5

            self.log.cyl_x = 0
            self.log.cyl_y = 0

        self.pub_image.publish(GRAY_FN)
        self.pub_rotation_velocity.publish(0)

        self.pub_cyl_radius.publish(0.5)
        self.pub_cyl_centre.publish(0,0,0)

        if (self.ratio_total > 2) and (old_id is not None):
            if self.condition in COOL_CONDITIONS:
                self.pub_pushover.publish("Fly %s flew %.1f loops (in %.1fs)" % (old_id, self.ratio_total, dt))
                self.pub_save.publish(old_id)

        self.update()

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

