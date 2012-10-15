import os.path
import time
import csv
import collections

import roslib; roslib.load_manifest('strawlab_freeflight_experiments')
import rospy
import std_msgs.msg

class CsvLogger:

    STATE = tuple()

    def __init__(self,fname=None, mode='w', directory="~/FLYDRA"):
        assert len(self.STATE)

        self._flydra_data_file = ''
        self._exp_uuid = ''

        for s in self.STATE:
            setattr(self, s, None)

        if fname is None:
            directory = os.path.join(
                            os.path.expanduser(directory),
                            rospy.get_name()[1:])
            if not os.path.exists(directory):
                os.makedirs(directory)
            self._fname = os.path.join(
                            directory,
                            time.strftime('DATA%Y%m%d_%H%M%S.csv'))
        else:
            self._fname = os.path.abspath(fname)

        if mode == 'r':
            rospy.loginfo("reading %s" % self._fname)
        elif mode == 'w':
            rospy.loginfo("writing %s" % self._fname)
            self._fd = open(self._fname,mode='w')
            self._fd.write(",".join(self.STATE))
            self._fd.write(",t_sec,t_nsec,flydra_data_file,exp_uuid\n")

            rospy.Subscriber('flydra_mainbrain_data_file',
                             std_msgs.msg.String,
                             self._on_flydra_mainbrain_start_saving)
            rospy.Subscriber('experiment/uuid',
                             std_msgs.msg.String,
                             self._on_experiment_uuid)
        else:
            raise Exception("mode must be 'r' or 'w'")

    def _on_flydra_mainbrain_start_saving(self, msg):
        self._flydra_data_file = msg.data

    def _on_experiment_uuid(self, msg):
        self._exp_uuid = msg.data

    def update(self, check=False):
        vals = [getattr(self,s) for s in self.STATE]

        if check and None in vals:
            rospy.logwarn("no state to save")

        self._fd.write(",".join(map(str,vals)))
        t = rospy.get_rostime()
        self._fd.write(",%d,%d,%s,%s\n" % (t.secs,t.nsecs,self._flydra_data_file,self._exp_uuid))
        self._fd.flush()

    def record_iterator(self):
        with open(self._fname, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            klass = None
            for row in reader:
                if klass is None:
                    klass = collections.namedtuple(self.__class__.__name__, row)
                    assert hasattr(klass, 't_sec')
                    continue

                yield klass(*row)

