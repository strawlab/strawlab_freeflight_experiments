#!/usr/bin/env python

import roslib
roslib.load_manifest('flyvr')

import rospy
import geometry_msgs.msg
import std_msgs.msg

import os
import argparse

import numpy as np
import scipy.misc

import flyvr.srv
import flyvr.display_client as display_client

def show_osgfile(fname,origin,scale):
    rospy.init_node('show_osgfile')

    mode_pub = display_client.DisplayServerProxy.set_stimulus_mode('StimulusOSGFile')

    scale_pub = rospy.Publisher("model_scale", geometry_msgs.msg.Vector3, latch=True, tcp_nodelay=True)
    centre_pub = rospy.Publisher("model_pose", geometry_msgs.msg.Pose, latch=True, tcp_nodelay=True)
    osg_pub = rospy.Publisher("stimulus_filename", std_msgs.msg.String, latch=True, tcp_nodelay=True)

    osg_pub.publish(fname)

    if origin is not None:
        xyz = map(float,origin.split(','))
        msg = geometry_msgs.msg.Pose()
        msg.position.x = xyz[0]
        msg.position.y = xyz[1]
        msg.position.z = xyz[2]
        msg.orientation.w = 1.0
        centre_pub.publish(msg)

    if scale is not None:
        print scale
        xyz = map(float,scale.split(','))
        msg = geometry_msgs.msg.Vector3(*xyz)
        scale_pub.publish(msg)

    rospy.spin()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fname',nargs=1,metavar="foo.osg")
    parser.add_argument('--origin', help='osg file origin (m) x,y,z', default=None)
    parser.add_argument('--scale', help='osg file scale (m) x,y,z', default=None)

    argv = rospy.myargv()
    args = parser.parse_args(argv[1:])

    show_osgfile(args.fname[0], args.origin, args.scale)


if __name__=='__main__':
    main()
