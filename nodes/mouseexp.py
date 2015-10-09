import time

import roslib
roslib.load_manifest('rospy')

import rospy
import std_msgs.msg

class Node(object):
    def __init__(self, period_sec=300, cycles=1):
        rospy.init_node('mousenode')
        self._pub_key = rospy.Publisher('/key', std_msgs.msg.String)
        self._pub_orientation = rospy.Publisher('/fear_orientation_deg', std_msgs.msg.Float32)

        self._period_sec = period_sec
        self._total_deg = cycles*360.0

    def _move_floor(self, event):
        t1 = time.time()
        pct = (t1-self.t0) / self._period_sec
        angle = pct * self._total_deg
        self._pub_orientation.publish(angle)

    def start_experiment(self):
        time.sleep(1)
        self._pub_key.publish('b')
        time.sleep(1)
        self._pub_key.publish('r')
        time.sleep(1)
        self.t0 = time.time()
        self.timer = rospy.Timer(rospy.Duration(0.1), self._move_floor)

        
if __name__ == "__main__":
    print "DID YOU START ROSBAG RECORD??"
    n = Node()
    n.start_experiment()
    rospy.spin()
