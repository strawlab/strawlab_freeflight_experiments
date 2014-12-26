import os.path
import time
import csv
import collections
import tempfile
import threading

import roslib; roslib.load_manifest('strawlab_freeflight_experiments')

import rospy
import std_msgs.msg
import strawlab_freeflight_experiments.conditions

class NoDataError(Exception):
    pass

class CsvLogger(object):

    STATE =         tuple()
    EXTRA_STATE =   ("t_sec","t_nsec","flydra_data_file","exp_uuid")
    FLY_STATE =     ("condition","lock_object","framenumber")
    DEFAULT_DIRECTORY = "~/FLYDRA"

    def __init__(self,fname=None, mode='w', directory=None, wait=False, use_tmpdir=False, continue_existing=None, state=None, use_rostime=False, warn=True, debug=True):

        if directory is None:
            directory = self.DEFAULT_DIRECTORY
        if use_tmpdir:
            directory = tempfile.mkdtemp()

        self._wlock = threading.Lock()
        self._flydra_data_file = ''
        self._exp_uuid = ''
        self._enable_warn = warn
        self._enable_debug = debug

        self._condition = ''
        self._pub_condition = None

        self._use_rostime = use_rostime

        self.last_tsecs = 0
        self.last_tnsecs = 0

        if mode not in ('r','w'):
            raise IOError("mode must be 'r' or 'w'. to continue an existing file set continue_existing=/PATH/TO/FILE")

        if continue_existing:
            if not os.path.isfile(continue_existing):
                raise NoDataError("file does not exist")

            fname = continue_existing
            mode = 'a'

        if fname is None:
            directory = os.path.expanduser(directory)
            if not os.path.exists(directory):
                os.makedirs(directory)
            self._fname = os.path.join(
                            directory,
                            time.strftime('%Y%m%d_%H%M%S') + '.%s.csv' % rospy.get_name()[1:])
        else:
            self._fname = os.path.abspath(fname)

        if mode in ('w','a'):
            #record_iterator returns all defined cols, so there is no need in requiring the
            #user pass state here.

            #backwards compat code to keep the same csv colum order and protect 
            #from duplicate columns
            if state is None:
                state = list(self.STATE)

            if not state:
                self._warn("you have not defined any additional state to save in the CSV file")

            self._state = []
            for s in state:
                if (s not in self._state) and (s not in self.FLY_STATE) and (s not in self.EXTRA_STATE):
                    self._state.append(s)

            for e in self.FLY_STATE:
                if e not in self._state:
                    self._state.append(e)

            self._cols = list(self._state)
            for e in self.EXTRA_STATE:
                if e not in self._cols:
                    self._cols.append(e)

            #set default values for properties
            for s in self._cols:
                setattr(self, s, None)

        if mode == 'r':
            self._debug("reading %s" % self._fname, to_ros=False)
        elif mode == 'w':
            self._debug("writing %s" % self._fname)
            if self._fname.startswith(tempfile.gettempdir()):
                self._warn("SAVING DATA TO TEMPORARY DIRECTORY - ARE YOU SURE")
            self._fd = open(self._fname,mode='w')
            self._fd.write(",".join(self._state))
            self._fd.write(",t_sec,t_nsec,flydra_data_file,exp_uuid\n")
        elif mode == 'a':
            self._debug("continuing %s" % self._fname)
            self._fd = open(self._fname,mode='a')

        if mode in ('w','a'):
            rospy.Subscriber('flydra_mainbrain/data_file',
                             std_msgs.msg.String,
                             self._on_flydra_mainbrain_start_saving)

            self._pub_condition_s = rospy.Publisher('condition_slash', std_msgs.msg.String)
            self._pub_condition_y = rospy.Publisher('condition_yaml', std_msgs.msg.String)
            self._pub_condition_n = rospy.Publisher('condition_name', std_msgs.msg.String)

            if wait:
                self._debug("waiting for flydra_mainbrain/data_file")
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

    def _warn(self, m, to_ros=True):
        if self._enable_warn:
            if to_ros:
                rospy.logwarn(m)
            else:
                print m

    def _debug(self, m, to_ros=True):
        if self._enable_debug:
            if to_ros:
                rospy.loginfo(m)
            else:
                print m

    def _on_flydra_mainbrain_start_saving(self, msg):
        self._flydra_data_file = msg.data

    def set_experiment_uuid(self, uuid):
        self._exp_uuid = uuid

    @property
    def condition(self):
        return self._condition

    @condition.setter
    def condition(self, v):
        if v is None:
            return

        if not isinstance(v, strawlab_freeflight_experiments.conditions.Condition):
            raise ValueError("YOU MUST ASSIGN CONDITION OBJECT TO LOG")

        s = v.to_slash_separated()
        y = v.to_yaml()
        n = v.name

        #still write the slash separated to the csv
        self._condition = s

        for val,pub in zip((s,y,n),(self._pub_condition_s,self._pub_condition_y,self._pub_condition_n)):
            if pub is not None:
                try:
                    pub.publish(val)
                except rospy.ROSException:
                    #node not yet initialized, which is usually just in test cases,
                    #in any case, this is not fatal even in practice and publishing
                    #condition is only for the operator-console anyway
                    pass

    @property
    def filename(self):
        return self._fname

    @property
    def columns(self):
        return self._cols

    def _update(self, check=False):
        if self._use_rostime:
            t = rospy.get_rostime()
            tsecs = t.secs
            tnsecs = t.nsecs
        else:
            #taken from ros.rostime (so the calculation is identical). we
            #never use the ros clock support anyway
            float_secs = time.time()
            tsecs = int(float_secs)
            tnsecs = int((float_secs - tsecs) * 1000000000)

        self.last_tsecs = tsecs
        self.last_tnsecs = tnsecs

        vals = [getattr(self,s) for s in self._state]

        if check and None in vals:
            self._warn("no state to save")

        self._fd.write(",".join(map(str,vals)))
        self._fd.write(",%d,%d,%s,%s\n" % (tsecs,tnsecs,self._flydra_data_file,self._exp_uuid))
        self._fd.flush()

    def update(self, check=False):
        with self._wlock:
            self._update(check)

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
                except TypeError, e:
                    self._warn("\ninvalid row: %r\n%s" % (row,e), to_ros=False)
                    continue

    def write_record(self, obj):
        self._fd.write(",".join([str(getattr(obj,s)) for s in self.columns]))
        self._fd.write("\n")

