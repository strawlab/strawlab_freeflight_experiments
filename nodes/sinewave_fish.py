#!/usr/bin/env python

import math
import numpy as np
import threading
import random

PACKAGE='strawlab_freeflight_experiments'

import roslib
import roslib.packages
roslib.load_manifest(PACKAGE)

import rospy
import flyvr.display_client as display_client
from std_msgs.msg import Int32, UInt32, Bool
from geometry_msgs.msg import Vector3
from ros_flydra.msg import flydra_mainbrain_super_packet
from strawlab_freeflight_experiments.msg import CylinderGratingInfo

import nodelib.node

from ros_flydra.constants import IMPOSSIBLE_OBJ_ID

pkg_dir = roslib.packages.get_pkg_dir(PACKAGE)

CONTROL_RATE        = 40.0      #Hz

FLY_DIST_CHECK_TIME = 5.0
FLY_DIST_MIN_DIST   = 0.2

TIMEOUT             = 0.5

class Node(nodelib.node.Experiment):
    def __init__(self, args):
        super(Node, self).__init__(args=args,
                                   state=("phase_velocity",))

        self._pub_stim_mode = display_client.DisplayServerProxy.set_stimulus_mode(
            'StimulusCylinderGrating')

        self.pub_geom_type = rospy.Publisher('grating_geometry_type',  Int32, latch=True, tcp_nodelay=True)
        self.pub_lock_z = rospy.Publisher('grating_lock_z', Bool, latch=True, tcp_nodelay=True)
        self.pub_grating_info = rospy.Publisher('grating_info', CylinderGratingInfo, latch=True, tcp_nodelay=True)

        self.pub_lock_object = rospy.Publisher('lock_object', UInt32, latch=True, tcp_nodelay=True)
        self.pub_lock_object.publish(IMPOSSIBLE_OBJ_ID)

        #protect the tracked id and fly position between the time syncronous main loop and the asyn
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

        self.switch_conditions()

        rospy.Subscriber("flydra_mainbrain/super_packets",
                         flydra_mainbrain_super_packet,
                         self.on_flydra_mainbrain_super_packets)

    def switch_conditions(self):

        #extract all the cyl grating params
        KEYS = ('phase_position', 'phase_velocity', 'wavelength', 'contrast', 'orientation')
        self.grating_conf = {k:float(self.condition[k]) for k in KEYS}

        self.drop_lock_on('switch conditions')

        self.pub_geom_type.publish(int(self.condition['geom_type']))
        self.pub_lock_z.publish(bool(self.condition['lock_z']))

        self.trigger_radius_start = float(self.condition['trigger_radius_start'])
        self.trigger_radius_stop = float(self.condition['trigger_radius_stop'])

        self.trigger_radius_stop = float(self.condition['trigger_radius_stop'])

        rospy.loginfo('condition: %s' % self.condition.name)


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
                    self.drop_lock_on('timeout')
                    continue

                if self.is_close_to_wall(fly_x,fly_y,fly_z):
                    self.drop_lock_on('close to wall')
                    continue

                if np.isnan(fly_x):
                    #we have a race  - a fly to track with no pose yet
                    continue

                active = True

#                #distance accounting, give up on fly if it is not moving
#                self.fly_dist += math.sqrt((fly_x-self.last_fly_x)**2 +
#                                           (fly_y-self.last_fly_y)**2 +
#                                           (fly_z-self.last_fly_z)**2)
#                self.last_fly_x = fly_x; self.last_fly_y = fly_y; self.last_fly_z = fly_z;

#                # drop slow moving flies
#                if now-self.last_check_flying_time > FLY_DIST_CHECK_TIME:
#                    fly_dist = self.fly_dist
#                    self.last_check_flying_time = now
#                    self.fly_dist = 0
#                    if fly_dist < FLY_DIST_MIN_DIST: # drop fly if it does not move enough
#                        self.drop_lock_on('slow')
#                        continue

            #new combine needs data recorded at the framerate
            self.log.framenumber = framenumber
            self.log.update()

            r.sleep()

        rospy.loginfo('%s finished. saved data to %s' % (rospy.get_name(), self.log.close()))

    def _is_in_sphere(self, pt3d, sphere, r):
        x,y,z = map(float,pt3d)
        cx,cy,cz = sphere
        return (x-cx)**2 + (y-cy)**2 + (z-cz)**2 < r**2

    def is_close_to_wall(self,px,py,pz):
        in_sphere = self._is_in_sphere((px,py,pz), (0,0,0.110), self.trigger_radius_stop)
        return (not in_sphere) and (pz < 0)

    def is_in_trigger_volume(self,pos):
        #trigger in a cylindrical column extending down to the wall
        in_cylinder = self._is_in_sphere((pos.x,pos.y,0),(0,0,0), self.trigger_radius_start)
        return in_cylinder and (pos.z < 0) and (not self.is_close_to_wall(pos.x,pos.y,pos.z))

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

    def _get_grating_msg(self, phase_multiplier):
        grating_conf = self.grating_conf.copy()
        phase_velocity = grating_conf['phase_velocity'] * phase_multiplier
        grating_conf['phase_velocity'] = phase_velocity
        msg = CylinderGratingInfo(**grating_conf)
        return phase_velocity, msg


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
            self.fly = obj.position
        
        phase_velocity, msg = self._get_grating_msg( random.choice((-1.0,1.0)) )
        self.pub_grating_info.publish(msg)
        self.log.phase_velocity = phase_velocity

        self.update()

    def drop_lock_on(self, reason=''):
        with self.trackinglock:
            old_id = self.currently_locked_obj_id
            now = rospy.get_time()
            dt = now - self.first_seen_time

            rospy.loginfo('dropping locked object %s %s (tracked for %.1f)' % (old_id,reason,dt))

            self.currently_locked_obj_id = None

            self.log.lock_object = IMPOSSIBLE_OBJ_ID
            self.log.framenumber = 0

        phase_velocity, msg = self._get_grating_msg(0.0)
        self.pub_grating_info.publish(msg)
        self.log.phase_velocity = phase_velocity

        self.update()        

def main():
    rospy.init_node("confinement")
    parser, args = nodelib.node.get_and_parse_commandline()
    node = Node(args)
    return node.run()

if __name__=='__main__':
    main()
