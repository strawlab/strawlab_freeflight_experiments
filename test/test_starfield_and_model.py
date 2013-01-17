#!/usr/bin/env python
import os
import sys
import time
import unittest
import shutil
import tempfile

PKG = 'strawlab_freeflight_experiments'
NAME = 'test_starfield_and_model'
import roslib; roslib.load_manifest(PKG)

import rospy
import rostest
from flyvr.msg import ROSPath

class TestStarfieldAndModel(unittest.TestCase):

    def __init__(self, *args):
        super(TestStarfieldAndModel, self).__init__(*args)
        self.success = False
        rospy.init_node(NAME, anonymous=True)

        topic_name = 'capture_frame_to_path'
        if rospy.resolve_name(topic_name) == ('/'+topic_name):
            rospy.logwarn("%s: topic %r has not been remapped! Typical command-line usage:\n"
                     "\t$ ./%s %s:=<%s topic>"%(NAME, topic_name, NAME, topic_name, topic_name))

        # request image from server
        frame_pub = rospy.Publisher(topic_name,
                                    ROSPath,
                                    latch=True)
        self.newdir = tempfile.mkdtemp()
        self.fname = os.path.join(self.newdir,NAME+'.png')
        assert not os.path.exists(self.fname)
        frame_pub.publish(self.fname)

    def test_geometry_image(self):
        timeout_t = time.time() + 10.0 #10 seconds
        while not rospy.is_shutdown() and not self.success and time.time() < timeout_t:
            time.sleep(0.1)
            # wait for new frame to be saved
            if os.path.exists(self.fname):
                # TODO: check that the image is actually valid and makes sense
                self.success = True
        shutil.rmtree(self.newdir)
        self.assert_(self.success, str(self.success))

if __name__ == '__main__':
    rostest.rosrun(PKG, NAME, TestStarfieldAndModel, sys.argv)

