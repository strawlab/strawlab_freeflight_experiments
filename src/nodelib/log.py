import os.path
import time

import roslib; roslib.load_manifest('rospy')
import rospy

class CsvLogger:

    STATE = tuple()

    def __init__(self,fname=None):
        assert len(self.STATE)
        if fname is None:
            fname = time.strftime('DATA%Y%m%d_%H%M%S.csv')
        self._fd = open(fname,mode='w')
        for s in self.STATE:
            setattr(self, s, None)
        self._fd.write(",".join(self.STATE))
        self._fd.write(",t_sec,t_nsec\n")
        rospy.loginfo("saving to %s" % fname)

    def update(self, check=False):
        vals = [getattr(self,s) for s in self.STATE]

        if check and None in vals:
            rospy.logwarn("no state to save")

        self._fd.write(",".join(map(str,vals)))
        t = rospy.get_rostime()
        self._fd.write(",%d,%d\n" % (t.secs,t.nsecs))
        self._fd.flush()
