#!/usr/bin/env python

import math
import numpy as np
import threading
import argparse

PACKAGE='strawlab_freeflight_experiments'

import roslib
import roslib.packages
roslib.load_manifest(PACKAGE)

import rospy
import display_client
from std_msgs.msg import UInt32, Bool, Float32, String
from geometry_msgs.msg import Vector3, Pose
from ros_flydra.msg import flydra_mainbrain_super_packet
import rospkg

import flyflypath.transform
import nodelib.log

pkg_dir = roslib.packages.get_pkg_dir(PACKAGE)

TOPIC_CYL_ROTATION      = "cylinder_rotation"
TOPIC_CYL_ROTATION_RATE = "cylinder_rotation_rate"
TOPIC_CYL_IMAGE         = "cylinder_image"

CONTROL_RATE        = 40.0      #Hz
SWITCH_MODE_TIME    = 5.0*60    #alternate between control and static (i.e. experimental control) seconds

FLY_DIST_CHECK_TIME = 5.0
FLY_DIST_MIN_DIST   = 0.2

START_RADIUS    = 0.12
START_ZDIST     = 0.4
START_Z         = 0.5

GRAY_FN = "gray.png"

TIMEOUT             = 0.5
IMPOSSIBLE_OBJ_ID   = 0

#CONDITIONS = ["checkerboard.png//+100.0", "gray.png//+100.0", "lena.png//+100.0"]
CONDITIONS = ["checkerboard.png//+100.0", "checkerboard.png//+200.0", "checkerboard.png//-100.0", "gray.png//+100.0"]
START_CONDITION = CONDITIONS[0]

XFORM = flyflypath.transform.SVGTransform()

class Logger(nodelib.log.CsvLogger):
    STATE = ("condition","rotation_rate","trg_x","trg_y","trg_z","lock_object","framenumber")

class Node(object):
    def __init__(self, wait_for_flydra, use_tmpdir):

        self._pub_stim_mode = display_client.DisplayServerProxy.set_stimulus_mode(
            'StimulusCylinder')

        self.rotation_pub = rospy.Publisher(TOPIC_CYL_ROTATION, Float32, latch=True, tcp_nodelay=True)
        self.rotation_pub.publish(0)
        self.rotation_velocity_pub = rospy.Publisher(TOPIC_CYL_ROTATION_RATE, Float32, latch=True, tcp_nodelay=True)
        self.image_pub = rospy.Publisher(TOPIC_CYL_IMAGE, String, latch=True, tcp_nodelay=True)

        self.lock_object = rospy.Publisher('lock_object', UInt32, latch=True, tcp_nodelay=True)
        self.lock_object.publish(IMPOSSIBLE_OBJ_ID)

        self.log = Logger(wait=wait_for_flydra, use_tmpdir=use_tmpdir)

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

        #start criteria for experiment
        self.x0 = self.y0 = 0
        #target (for moving points)
        self.trg_x = self.trg_y = 0.0

        self.svg_pub = rospy.Publisher("svg_filename", String)
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

        img,svg,p = self.condition.split('/')
        self.img_fn = str(img)
        self.svg_fn = str(svg)
        self.p_const = float(p)

        if self.svg_fn:
            self.svg_pub.publish(self.svg_fn)
        
        rospy.loginfo('condition: %s (p=%f, svg=%s)' % (self.condition,self.p_const,self.svg_fn))

    def get_rotation_velocity_vector(self,fly_x,fly_y,fly_z, fly_vx, fly_vy, fly_vz):
        #px,py = XFORM.xy_to_pxpy(fly_x,fly_y)
        #vx,vy = XFORM.xy_to_pxpy(fly_vx, fly_vy)

        dpos = np.array((fly_x-self.x0,fly_y-self.y0))
        vel  = np.array((fly_vx, fly_vy))
        magn = np.cross(dpos,vel)

        return magn*self.p_const,self.x0,self.y0

    def run(self):
        rospy.loginfo('running stimulus')
        r = rospy.Rate(CONTROL_RATE)
        while not rospy.is_shutdown():
            with self.trackinglock:
                currently_locked_obj_id = self.currently_locked_obj_id
                fly_x = self.fly.x; fly_y = self.fly.y; fly_z = self.fly.z
                fly_vx = self.flyv.x; fly_vy = self.flyv.y; fly_vz = self.flyv.z

            if currently_locked_obj_id is None:
                rate = 0
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

                px,py = XFORM.xy_to_pxpy(fly_x,fly_y)
                self.src_pub.publish(px,py,0)
                trg_px, trg_py = XFORM.xy_to_pxpy(trg_x,trg_y)
                self.trg_pub.publish(trg_px,trg_py,0)

                self.log.rotation_rate = rate
                self.log.trg_x = trg_x; self.log.trg_y = trg_y; self.log.trg_z = START_Z
                self.log.update()

            self.rotation_velocity_pub.publish(rate)
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
        self.lock_object.publish( self.log.lock_object )

    def lock_on(self,obj,framenumber):
        rospy.loginfo('locked object %d at frame %d' % (obj.obj_id,framenumber))
        now = rospy.get_time()
        self.currently_locked_obj_id = obj.obj_id
        self.last_seen_time = now
        self.log.lock_object = obj.obj_id
        self.log.framenumber = framenumber
        self.image_pub.publish( self.img_fn )
        self.update()

    def drop_lock_on(self):
        rospy.loginfo('dropping locked object %s' % self.currently_locked_obj_id)
        self.currently_locked_obj_id = None
        self.log.lock_object = IMPOSSIBLE_OBJ_ID
        self.log.framenumber = 0
        self.image_pub.publish( GRAY_FN )
        self.update()

def main():
    rospy.init_node("rotation")

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

