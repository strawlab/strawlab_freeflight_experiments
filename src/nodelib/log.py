import os.path
import time
import csv
import collections
import tempfile

import roslib; roslib.load_manifest('strawlab_freeflight_experiments')
import rospy
import std_msgs.msg

class NoDataError(Exception):
    pass

class CsvLogger:

    STATE = tuple()
    EXTRA_STATE = ("t_sec","t_nsec","flydra_data_file","exp_uuid")
    DEFAULT_DIRECTORY = "~/FLYDRA"

    def __init__(self,fname=None, mode='w', directory=None, wait=False, use_tmpdir=False):
        assert len(self.STATE)

        if directory is None:
            directory = self.DEFAULT_DIRECTORY
        if use_tmpdir:
            directory = tempfile.mkdtemp()

        self._flydra_data_file = ''
        self._exp_uuid = ''

        self._cols = list(self.STATE)
        self._cols.extend(self.EXTRA_STATE)

        for s in self._cols:
            setattr(self, s, None)

        if fname is None:
            directory = os.path.expanduser(directory)
            if not os.path.exists(directory):
                os.makedirs(directory)
            self._fname = os.path.join(
                            directory,
                            time.strftime('%Y%m%d_%H%M%S') + '.%s.csv' % rospy.get_name()[1:])
        else:
            self._fname = os.path.abspath(fname)

        if mode == 'r':
            rospy.loginfo("reading %s" % self._fname)
        elif mode == 'w':
            rospy.loginfo("writing %s" % self._fname)
            if self._fname.startswith(tempfile.gettempdir()):
                rospy.logwarn("SAVING DATA TO TEMPORARY DIRECTORY - ARE YOU SURE")
            self._fd = open(self._fname,mode='w')
            self._fd.write(",".join(self.STATE))
            self._fd.write(",t_sec,t_nsec,flydra_data_file,exp_uuid\n")

            rospy.Subscriber('flydra_mainbrain/data_file',
                             std_msgs.msg.String,
                             self._on_flydra_mainbrain_start_saving)
            rospy.Subscriber('experiment_uuid',
                             std_msgs.msg.String,
                             self._on_experiment_uuid)

            if wait:
                rospy.loginfo("waiting for flydra_mainbrain/data_file")
                t0 = t1 = rospy.get_time()
                while (t1 - t0) < 5.0: #seconds
                    rospy.sleep(0.1)
                    t1 = rospy.get_time()
                    if self._flydra_data_file:
                        break
                if not self._flydra_data_file:
                    self.close()
                    os.remove(self._fname)
                    raise NoDataError
        else:
            raise IOError("mode must be 'r' or 'w'")

    def _on_flydra_mainbrain_start_saving(self, msg):
        self._flydra_data_file = msg.data

    def _on_experiment_uuid(self, msg):
        self._exp_uuid = msg.data

    @property
    def filename(self):
        return self._fname

    @property
    def columns(self):
        return self._cols

    def update(self, check=False):
        vals = [getattr(self,s) for s in self.STATE]

        if check and None in vals:
            rospy.logwarn("no state to save")

        self._fd.write(",".join(map(str,vals)))
        t = rospy.get_rostime()
        self._fd.write(",%d,%d,%s,%s\n" % (t.secs,t.nsecs,self._flydra_data_file,self._exp_uuid))
        self._fd.flush()

    def close(self):
        self._fd.close()
        return self._fname

    def record_iterator(self):
        with open(self._fname, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            klass = None
            for row in reader:
                if klass is None:
                    klass = collections.namedtuple(self.__class__.__name__, row)
                    assert hasattr(klass, 't_sec')
                    continue

                try:
                    yield klass(*row)
                except TypeError:
                    rospy.logwarn("invalid row: %r" % row)
                    continue

    def write_record(self, obj):
        self._fd.write(",".join([str(getattr(obj,s)) for s in self.columns]))
        self._fd.write("\n")

