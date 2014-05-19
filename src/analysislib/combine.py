import os.path
import sys
import argparse
import Queue
import random
import time
import cPickle as pickle
import re

import tables
import pandas as pd
import numpy as np
import pytz
import datetime
import calendar

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')

import autodata.files
import nodelib.log

import analysislib.fixes
import analysislib.filters
import analysislib.combine
import analysislib.args
import analysislib.curvature as acurve
import analysislib.plots as aplt

from ros_flydra.constants import IMPOSSIBLE_OBJ_ID, IMPOSSIBLE_OBJ_ID_ZERO_POSE
from strawlab.constants import DATE_FMT

#results = {
#   condition:{
#       df:[dataframe,...],
#       start_obj_ids:[(x0,y0,obj_id,framenumber0,time0),...]
#       count:[n_frames,...],
#   }
#}

def safe_condition_string(s):
    return "".join([c for c in s if re.match(r'\w', c)])

class _Combine(object):

    calc_linear_stats = True
    calc_angular_stats = True
    calc_turn_stats = True

    def __init__(self, **kwargs):
        self.__debug = kwargs.get("debug",True)
        self.__warn = kwargs.get("warn",True)
        self._dt = None
        self._lenfilt = None
        self._idfilt = []
        self._skipped = {}
        self._results = {}
        self._custom_filter = None
        self._custom_filter_min = None
        self._tzname = 'Europe/Vienna'
        self._tfilt_before = None
        self._tfilt_after = None
        self._tfilt = None
        self._plotdir = None

    def _get_trajectories(self, h5):
        trajectories = h5.root.trajectories

        #unexplainable protip - adding an index on the framenumber table makes
        #things slloooowwww
        if trajectories.cols.framenumber.is_indexed:
            trajectories.cols.framenumber.removeIndex()

        if not trajectories.cols.obj_id.is_indexed:
            try:
                trajectories.cols.obj_id.createIndex()
            except tables.exceptions.FileModeError:
                self._warn("obj_id column not indexed, this will be slow. reindex")

        return trajectories

    def _maybe_apply_tfilt_should_save(self, ts):
        #returns true to indicate the data should be saved
        if (self._tfilt_before is None) and (self._tfilt_after is None):
            return True

        #now we should have parsed an h5 file, so can actually create a
        #timestamp in utc from the timezone of the data and the datetime spec
        #as given
        if self._tfilt is None:
            spec = self._tfilt_before if self._tfilt_before is not None else self._tfilt_after
            lt = self.timezone.localize(spec)
            self._tfilt = calendar.timegm(lt.utctimetuple())

        if self._tfilt_before is not None:
            return ts < self._tfilt
        else:
            return ts > self._tfilt

    def _maybe_add_tfilt(self, args):
        for f in ("tfilt_before", "tfilt_after"):
            v = getattr(args,f,None)
            if v is not None:
                setattr(self, "_%s" % f, datetime.datetime.strptime(v, DATE_FMT))

    def _maybe_add_customfilt(self, args):
        if args.customfilt is not None and args.customfilt_len is not None:
            self.add_custom_filter(args.customfilt, args.customfilt_len)

    def _debug(self, m):
        if self.__debug:
            print m

    def _warn(self, m):
        if self.__warn:
            print m

    @property
    def plotdir(self):
        """the directory in which to store plots for this analysis"""
        if self._plotdir is None:
            return self._plotdir

        me = os.path.basename(sys.argv[0])
        pd = os.path.join(self._plotdir, me)
 
        if not os.path.isdir(pd):
            os.makedirs(pd)

        return pd

    @plotdir.setter
    def plotdir(self, val):
        self._plotdir = val

    @property
    def timezone(self):
        """the timezone in which the experiment was started"""
        return pytz.timezone(self._tzname)

    @property
    def fname(self):
        return self.get_plot_filename(
                        os.path.basename(self.csv_file).split('.')[0])

    @property
    def min_num_frames(self):
        """the minimum number of frames for the given lenfilt and dt"""
        try:
            return self._lenfilt / self._dt
        except TypeError:
            return 1

    @property
    def custom_filter_min_num_frames(self):
        """the minimum number of frames for the given customfilter and dt"""
        try:
            return self._custom_filter_min / self._dt
        except TypeError:
            return 1

    @property
    def framerate(self):
        """the framerate of the data"""
        return 1.0 / self._dt

    def get_plot_filename(self, name):
        """return a full path to the autodata directory to save any plots"""
        return os.path.join(self.plotdir,name)

    def get_num_skipped(self, condition):
        """returns the number of skipped trials for the given condition.

        trials are skipped if they do not meet all the filter criteria"""
        return self._skipped.get(condition,0)

    def get_num_analysed(self, condition):
        """returns the number of trials which met the filter criteria"""
        return self._results[condition]['count']

    def get_num_trials(self, condition):
        """returns the total number of trials in the data for the givent condition"""
        return self.get_num_skipped(condition) + self.get_num_analysed(condition)

    def get_total_trials(self):
        """returns the total trials for all conditions"""
        return sum([self.get_num_trials(c) for c in self.get_conditions()])

    def get_num_conditions(self):
        """returns the number of experimental conditions"""
        return len(self._results)

    def get_conditions(self):
        """returns a list of the names of the experimental conditions"""
        return self._results.keys()

    def get_num_frames(self, seconds):
        """returns the number of frames that should be recorded for the given seconds"""
        return seconds / self._dt

    def get_spanned_results(self):
        """
        returns a dict of object ids whose trajectories span multiple conditions.
        the values in the dict are a list of 2-tuples
            (condition,nframes)
        """
        return {}

    def enable_debug(self):
        self.__debug = True

    def disable_debug(self):
        self.__debug = False

    def enable_warn(self):
        self.__warn = True

    def disable_warn(self):
        self.__warn = False


    def get_results(self):
        """Returns all data for all conditions that passed the configured filters

        Returns:
            A two-tuple.

            the first retured variable is a nested dict of the following 
            structure

            results = {
               condition:{
                   df:[dataframe,...],
                   start_obj_ids:[(x0,y0,obj_id,framenumber0,time0),...]
                   count:[n_frames,...],
               }

            The second is dt

        """
        return self._results, self._dt

    def get_one_result(self, obj_id):
        """
        Get the data associated with a single object_id

        returns: (dataframe, dt, (x0,y0,obj_id,framenumber0,time0))
        """
        if obj_id in self.get_spanned_results():
            raise ValueError("obj_id: %s exists in multiple conditions" % obj_id)

        for i,(current_condition,r) in enumerate(self._results.iteritems()):
            for df,(x0,y0,_obj_id,framenumber0,time0) in zip(r['df'], r['start_obj_ids']):
                if _obj_id == obj_id:
                    return df,self._dt,(x0,y0,obj_id,framenumber0,time0)

        raise ValueError("No such obj_id: %s" % obj_id)

    def get_result_columns(self):
        """get the names of the columns in the combined dataframe"""
        for current_condition,r in self._results.iteritems():
            for df in r['df']:
                return list(df.columns)
        return []

    def add_custom_filter(self, s, post_filter_min):
        if 'df[' not in s:
            raise Exception("incorrectly formatted filter string: %s" % s)
        if ' ' in s:
            raise Exception("incorrectly formatted filter string (containts spaces): %s" % s)
        if post_filter_min is None:
            raise Exception("filter minimum must be given")
        self._custom_filter = s
        self._custom_filter_min = post_filter_min

class _CombineFakeInfinity(_Combine):
    def __init__(self, **kwargs):
        _Combine.__init__(self, **kwargs)
        self._nconditions = kwargs.get('nconditions',1)
        self._ntrials = kwargs.get('ntrials', 100)

        self._results = {}
        self._dt = 1/100.0
        self._t0 = time.time()
        self._tzname = 'UTC'

    def _add(self):
        obj_id = 1
        framenumber = 1

        for c in range(self._nconditions):
            cond = "tex%d/svg/1.0/1.0/adv/..." % c

            try:
                self._results[cond]
            except KeyError:
                self._results[cond] = {"df":[],"start_obj_ids":[],"count":0}
                self._skipped[cond] = 0

            for t in range(self._ntrials):
                if obj_id == 1:
                    #make sure the first infinity is full and perfect
                    df = self.get_fake_infinity(5,0,0,0,framenumber,0,0,self._dt)
                else:
                    df = self.get_fake_infinity(
                                n_infinity=5,
                                random_stddev=0.008,
                                x_offset=0.1 * (random.random() - 0.5),
                                y_offset=0.1 * (random.random() - 0.5),
                                frame0=framenumber,
                                nan_pct=1,
                                latency=10,
                                dt=self._dt
                    )
                    #take some random last number of trajectory
                    df = df.tail(random.randrange(10,len(df)))

                if len(df) < self.min_num_frames:
                    self._skipped[cond] += 1
                    continue

                dt = self._dt
                if self.calc_linear_stats:
                    acurve.calc_velocities(df, dt)
                    acurve.calc_accelerations(df, dt)
                if self.calc_angular_stats:
                    acurve.calc_angular_velocities(df, dt)
                if self.calc_turn_stats:
                    acurve.calc_curvature(df, dt, 10, 'leastsq', clip=(0,1))

                if self._custom_filter is not None:
                    df = eval(self._custom_filter)
                    n_samples = len(df)
                    if n_samples < self.custom_filter_min_num_frames:
                        self._debug('FILTER: %d for obj_id %d' % (n_samples,obj_id))
                        self._skipped[cond] += 1
                        continue

                first = df.irow(0)
                last = df.irow(-1)
                f0 = first.name

                self._results[cond]["df"].append(df)
                self._results[cond]["count"] += 1
                self._results[cond]["start_obj_ids"].append(
                        (first['x'],first['y'],obj_id,f0,self._t0 + (f0 * self._dt))
                )

                obj_id += 1
                framenumber += len(df)

    @staticmethod
    def get_fake_infinity_trajectory(n_infinity, random_stddev, x_offset, y_offset, frame0):
        def get_noise(d):
            if random_stddev:
                return (np.random.random(len(d)) - 0.5) * random_stddev
            else:
                return np.zeros_like(d)

        pi = np.pi
        leaft = np.linspace(-pi/4.,pi/4., 100)
        leaftrev = (leaft-pi)[::-1] #reverse

        theta = np.concatenate( (leaft, leaftrev[1:]) )

        r = np.cos(2*theta)

        x = np.concatenate( ([0], r*np.cos( theta )) )
        y = np.concatenate( ([0], r*np.sin( theta )) )

        x *= 0.4
        y *= 0.8

        x += x_offset
        y += y_offset

        N = len(x)

        z = np.zeros_like(x)
        z[:] = 0.7

        ratio = np.array(range(0,N)) / float(N)

        if n_infinity > 1:
            df = pd.DataFrame({
                    "x":np.concatenate( [x + get_noise(x) for i in range(n_infinity) ] ),
                    "y":np.concatenate( [y + get_noise(y) for i in range(n_infinity) ] ),
                    "z":np.concatenate( [z + get_noise(z) for i in range(n_infinity) ] ),
                    "ratio":np.concatenate( [ratio + get_noise(ratio) for i in range(n_infinity) ] )},
                    index=range(frame0,frame0+(N*n_infinity))
            )
        else:
            df = pd.DataFrame({
                    "x":x + get_noise(x),
                    "y":y + get_noise(y),
                    "z":z + get_noise(z),
                    "ratio":ratio + get_noise(ratio)},
                    index=range(frame0,frame0+N)
            )

        return df

    @staticmethod
    def get_fake_infinity(n_infinity, random_stddev, x_offset, y_offset, frame0, nan_pct, latency, dt):
        df = _CombineFakeInfinity.get_fake_infinity_trajectory(n_infinity, random_stddev, x_offset, y_offset, frame0)

        #despite these being recomputed later, we need to get them first
        #to make sure rrate is correlated to dtheta, we remove the added colums later
        cols = []
        cols.extend( acurve.calc_velocities(df, dt) )
        cols.extend( acurve.calc_angular_velocities(df, dt) )

        #add some uncorrelated noise to rrate
        rrate = (df['dtheta'].values * 10.0) + (0.0 * (np.random.random(len(df)) - 0.5))

        if latency > 0:
            rrate = np.concatenate( (rrate[latency:],np.random.random(latency)) )

        #rotation rate comes from a csv file, with a lower freq, so trim some % of the data
        #to simulate missing values
        trim = np.random.random(len(rrate)) < (nan_pct / 100.0)
        rrate[trim] = np.nan

        df['rotation_rate'] = rrate
        df['v_offset_rate'] = np.zeros_like(rrate)

        for c in cols:
            del df[c]

        return df

    def add_from_args(self, args):
        self._lenfilt = args.lenfilt
        self._maybe_add_tfilt(args)
        self._maybe_add_customfilt(args)
        self.plotdir = args.outdir if args.outdir else os.getcwd()
        self.csv_file = "test"

        self._add()

    def add_test_infinities(self):
        self._lenfilt = 0
        self.plotdir = "/tmp"
        self.csv_file = "test"
        self._add()

class CombineCSV(_Combine):

    csv_file = ''

    def __init__(self, **kwargs):
        _Combine.__init__(self, **kwargs)
        #when we add multiple csv files we have to ensure the frame_number is
        #monotonically increasing. we are lucky that the convention for csv
        #files generated by the tethered rig is that lock_obj = int(time.time())
        self._last_framenumber = 0
        self._df = None
        self._csv_suffix = kwargs.get("csv_suffix")


    def add_from_args(self, args, csv_suffix=None):
        if not csv_suffix:
            csv_suffix = self._csv_suffix

        self._idfilt = args.idfilt
        self._maybe_add_tfilt(args)

        if args.uuid is not None:
            if len(args.uuid) > 1:
                self.plotdir = args.outdir

            for uuid in args.uuid:
                fm = autodata.files.FileModel(basedir=args.basedir)
                fm.select_uuid(uuid)
                csv_file = fm.get_file_model(csv_suffix).fullpath

                #this handles the single uuid case
                if self.plotdir is None:
                    self.plotdir = args.outdir if args.outdir else fm.get_plot_dir()

                self.add_csv_file(csv_file, args.lenfilt)
        else:
            self.add_csv_file(args.csv_file, args.lenfilt)
            self.plotdir = args.outdir if args.outdir else os.getcwd()

    def add_csv_file(self, csv_file, lenfilt=None):
        self._debug("IO:     reading %s" % csv_file)
        self.csv_file = csv_file

        df = pd.DataFrame.from_csv(self.csv_file,index_col="framenumber")
        assert 'lock_object' in df.columns

        df = df.fillna(method="pad")
        df['time'] = df['t_sec'] + (df['t_nsec'] * 1e-9)

        dt = (df['time'].values[-1] - df['time'].values[0]) / len(df)

        if self._dt is None:
            self._dt = dt
            self._lenfilt = lenfilt
        else:
            #check the new csv file was recorded with the same timebase
            assert abs(dt-self._dt) < 1e-4

        for cond,dfc in df.groupby('condition'):

            if cond not in self._results:
                self._results[cond] = {'df':[],'start_obj_ids':[],'count':0}

            for obj_id,dfo in dfc.groupby('lock_object'):
                if obj_id == 0:
                    continue

                if not self._df_ok(dfo):
                    continue

                if self._idfilt and (obj_id not in self._idfilt):
                    continue

                dt = self._dt
                if self.calc_linear_stats:
                    acurve.calc_velocities(dfo, dt)
                    acurve.calc_accelerations(dfo, dt)
                if self.calc_angular_stats:
                    acurve.calc_angular_velocities(dfo, dt)
                if self.calc_turn_stats:
                    acurve.calc_curvature(dfo, dt, 10, 'leastsq', clip=(0,1))

                self._results[cond]['df'].append(dfo)
                self._results[cond]['start_obj_ids'].append(self._get_result(dfo))
                self._results[cond]['count'] += 1


        if self._df is None:
            self._df = df
        else:
            last = df.last_valid_index() + self._last_framenumber + 1
            #increase the indexes of this df to be past the end of the last one
            df = df.set_index(df.index.values + last)
            #append / concat this at the end of the current dataframe
            self._df = pd.concat((self._df, df))
            self._last_framenumber = last

    def _df_ok(self, df):
        if len(df) < self.min_num_frames:
            return False
        else:
            return True

    def _get_result(self, df):
        ser = df.irow(0)
        return ser['x'],ser['y'],ser['lock_object'],ser.name,ser['time']

    def get_one_result(self, obj_id):
        df = self._df[self._df['lock_object'] == obj_id]
        if not len(df):
            raise ValueError("object %s not found" % obj_id)

        if not self._df_ok(df):
            raise ValueError("result is too short")

        return df,self._dt,self._get_result(df)

class CombineH5(_Combine):

    h5_file = ''

    def __init__(self, **kwargs):
        _Combine.__init__(self, **kwargs)
        self._dt = None

    def add_from_args(self, args):
        self._maybe_add_tfilt(args)
        self._maybe_add_customfilt(args)

        if args.uuid is not None:
            uuid = args.uuid[0]
            fm = autodata.files.FileModel(basedir=args.basedir)
            fm.select_uuid(uuid)
            h5_file = fm.get_file_model("simple_flydra.h5").fullpath
        else:
            h5_file = args.h5_file

        self.add_h5_file(h5_file)

    def add_from_uuid(self, uuid, *args, **kwargs):
        fm = autodata.files.FileModel()
        fm.select_uuid(uuid)
        h5_file = fm.get_file_model("simple_flydra.h5").fullpath
        self.add_h5_file(h5_file)

    def add_h5_file(self, h5_file):
        self._debug("IO:     reading %s" % h5_file)

        warnings = {}

        self.h5_file = h5_file

        h5 = tables.openFile(h5_file, mode='r')

        self._trajectories = self._get_trajectories(h5)
        dt = 1.0/self._trajectories.attrs['frames_per_second']

        self._trajectory_start_times = h5.root.trajectory_start_times
        tzname = h5.root.trajectory_start_times.attrs['timezone']

        if self._dt is None:
            self._dt = dt
            self._tzname = tzname
        else:
            assert dt == self._dt
            assert tzname == self._tzname

    def get_one_result(self, obj_id):
        query = "obj_id == %d" % obj_id

        traj = self._trajectories.readWhere(query)
        start = self._trajectory_start_times.readWhere(query)

        _,tsec,tnsec = start[0]
        t0 = tsec+(tnsec*1e-9)

        df = pd.DataFrame(
                {i:traj[i] for i in 'xyz'},
                index=traj['framenumber']
        )

        dt = self._dt
        if self.calc_linear_stats:
            acurve.calc_velocities(df, dt)
            acurve.calc_accelerations(df, dt)
        if self.calc_angular_stats:
            acurve.calc_angular_velocities(df, dt)
        if self.calc_turn_stats:
            acurve.calc_curvature(df, dt, 10, 'leastsq', clip=(0,1))

        return df,self._dt,(traj['x'][0],traj['y'][0],obj_id,traj['framenumber'][0],t0)

class CombineH5WithCSV(_Combine):
    """
    Combines the data from a smoothed (simple_flydra) h5 file with
    a csv file based on the framenumber
    """

    csv_file = ''
    h5_file = ''

    def __init__(self, *csv_cols, **kwargs):
        _Combine.__init__(self, **kwargs)
        cols = ["framenumber"]
        cols.extend(csv_cols)
        self._cols = set(cols)
        self._csv_suffix = kwargs.get("csv_suffix")

        #use this for keeping track of results that span multiple conditions
        self._results_by_condition = {}

    def add_from_uuid(self, uuid, csv_suffix=None, frames_before=0, **kwargs):
        """Add a csv and h5 file collected from the experiment with the
        given uuid
        """
        if not csv_suffix:
            csv_suffix = self._csv_suffix

        fm = autodata.files.FileModel()
        fm.select_uuid(uuid)
        csv_file = fm.get_file_model(csv_suffix).fullpath
        h5_file = fm.get_file_model("simple_flydra.h5").fullpath

        if 'args' in kwargs:
            args = kwargs['args']
        else:
            parser,args = analysislib.args.get_default_args()
            for k in kwargs:
                setattr(args,k,kwargs[k])

        #this handles the single uuid case
        if not self.plotdir:
            self.plotdir = args.outdir if args.outdir else fm.get_plot_dir()

        self.add_csv_and_h5_file(csv_file, h5_file, args, frames_before)

    def add_from_args(self, args, csv_suffix=None, frames_before=0):
        """Add possibly multiple csv and h5 files based on the command line
        arguments given
        """
        if not csv_suffix:
            csv_suffix = self._csv_suffix

        self._maybe_add_tfilt(args)
        self._maybe_add_customfilt(args)

        if args.uuid is not None:
            if len(args.uuid) > 1:
                self.plotdir = args.outdir

            for uuid in args.uuid:
                fm = autodata.files.FileModel(basedir=args.basedir)
                fm.select_uuid(uuid)
                csv_file = fm.get_file_model(csv_suffix).fullpath
                h5_file = fm.get_file_model("simple_flydra.h5").fullpath

                #this handles the single uuid case
                if self.plotdir is None:
                    self.plotdir = args.outdir if args.outdir else fm.get_plot_dir()

                self.add_csv_and_h5_file(csv_file, h5_file, args, frames_before)

        else:
            csv_file = args.csv_file
            h5_file = args.h5_file

            self.plotdir = args.outdir if args.outdir else os.getcwd()

            self.add_csv_and_h5_file(csv_file, h5_file, args, frames_before)

    def get_spanned_results(self):
        spanned = {}
        for oid,details in self._results_by_condition.iteritems():
            if len(details) > 1:
                spanned[oid] = details
        return spanned

    def add_csv_and_h5_file(self, csv_fname, h5_file, args, frames_before=0):
        """Add a single csv and h5 file"""

        warnings = {}

        self.csv_file = csv_fname
        self.h5_file = h5_file

        if args.cached:
            name = self.get_plot_filename("data.pkl")
            self._debug("IO:     reading %s" % name)
            if os.path.isfile(name):
                with open(name,'r+b') as f:
                    d = pickle.load(f)
                    self._results = d['results']
                    self._dt = d['dt']
                    return

        self._fix = analysislib.fixes.load_fixups(csv_file=os.path.basename(self.csv_file),
                                                   h5_file=os.path.basename(self.h5_file))

        self._debug("IO:     reading %s" % csv_fname)
        self._debug("IO:     reading %s" % h5_file)
        if self._fix.active:
            self._debug("IO:     fixing data %s" % self._fix)

        #record_iterator in the csv_file returns all defined cols by default.
        #those specified in csv_cols are float()'d and put into the dataframe
        infile = nodelib.log.CsvLogger(csv_fname, "r")

        h5 = tables.openFile(h5_file, mode='r+' if args.reindex else 'r')

        trajectories = self._get_trajectories(h5)
        dt = 1.0/trajectories.attrs['frames_per_second']

        tzname = h5.root.trajectory_start_times.attrs['timezone']

        if self._dt is None:
            self._dt = dt
            self._lenfilt = args.lenfilt
            self._tzname = tzname
        else:
            assert dt == self._dt
            assert tzname == self._tzname

        dur_samples = self.min_num_frames

        _ids = Queue.Queue(maxsize=2)
        this_id = IMPOSSIBLE_OBJ_ID
        this_cond = None
        csv_results = {}

        results = self._results
        this_row = {}

        skipped = self._skipped

        for r in infile.record_iterator():

            row = self._fix.fix_row(r)

            try:

                _cond = str(row.condition)
                _id = int(row.lock_object)
                _t = float(row.t_sec) + (float(row.t_nsec) * 1e-9)
                _framenumber = int(row.framenumber)

                for k in self._cols:
                    if k != "framenumber":
                        try:
                            this_row[k] = float(getattr(row,k))
                            warn = None
                        except AttributeError:
                            this_row[k] = np.nan
                            warn = "%s:col" % k, "no such column in csv:%s" % k
                            if k not in warnings:
                                self._warn()
                                warnings[k] = True
                        except ValueError:
                            this_row[k] = np.nan
                            warn = "%s:val" % k, "missing value for column in csv:%s" % k

                        if warn is not None:
                            warnk,warnv = warn
                            if warnk not in warnings:
                                self._warn("WARNING: %s" % warnv)
                                warnings[warnk] = True


                if _cond not in results:
                    results[_cond] = dict(count=0,
                                          start_obj_ids=[],
                                          df=[])
                    skipped[_cond] = 0
                    

                if _id == IMPOSSIBLE_OBJ_ID_ZERO_POSE:
                    continue
                if _id == IMPOSSIBLE_OBJ_ID:
                    continue
                elif (_id != this_id) or (_cond != this_cond):

                    try:
                        query_id,query_framenumber,start_time,query_cond = _ids.get(False)
                    except Queue.Empty:
                        #first time
                        this_id = _id
                        this_cond = _cond
                        csv_results = {k:[] for k in self._cols}
                        query_id = None
                    finally:
                        _ids.put((_id,_framenumber,_t,_cond),block=False)

                    #first time
                    if query_id is None:
                        continue

                    if (not args.idfilt) or (query_id in args.idfilt):

                        assert this_cond == query_cond

                        r = results[query_cond]

                        if frames_before != 0:
                            start_frame = query_framenumber - frames_before
                        else:
                            start_frame = query_framenumber

                        stop_frame = _framenumber

                        query = "(obj_id == %d) & (framenumber >= %d) & (framenumber < %d)" % (
                                        query_id,
                                        start_frame,
                                        stop_frame)

                        valid = trajectories.readWhere(query)

                        #filter the trajectories based on Z value
                        valid_z_cond = analysislib.filters.filter_z(
                                                    args.zfilt,
                                                    valid['z'],
                                                    args.zfilt_min, args.zfilt_max)
                        #filter based on radius
                        valid_r_cond = analysislib.filters.filter_radius(
                                                    args.rfilt,
                                                    valid['x'],valid['y'],
                                                    args.rfilt_max)

                        valid_cond = valid_z_cond & valid_r_cond

                        validx = valid['x'][valid_cond]
                        validy = valid['y'][valid_cond]
                        validz = valid['z'][valid_cond]
                        validframenumber = valid['framenumber'][valid_cond]

                        n_samples = len(validx)
                        df = None
                        if n_samples < dur_samples: # must be at least this long
                            self._debug('TRIM:   %d samples for obj_id %d' % (n_samples,query_id))
                            self._skipped[_cond] += 1
                        else:
                            dfd = {'x':validx,'y':validy,'z':validz}

                            for k in self._cols:
                                if k != "framenumber":
                                    dfd[k] = pd.Series(csv_results[k],index=csv_results['framenumber'])

                            df = pd.DataFrame(dfd,index=validframenumber)

                            dt = self._dt
                            if self.calc_linear_stats:
                                acurve.calc_velocities(df, dt)
                                acurve.calc_accelerations(df, dt)
                            if self.calc_angular_stats:
                                acurve.calc_angular_velocities(df, dt)
                            if self.calc_turn_stats:
                                acurve.calc_curvature(df, dt, 10, 'leastsq', clip=(0,1))

                            if self._custom_filter is not None:
                                df = eval(self._custom_filter)
                                n_samples = len(df)
                                if n_samples < self.custom_filter_min_num_frames:
                                    self._debug('FILTER: %d for obj_id %d' % (n_samples,query_id))
                                    self._skipped[_cond] += 1
                                    df = None

                            if not self._maybe_apply_tfilt_should_save(start_time):
                                df = None

                        if df is not None:

                            if _cond != this_cond:
                                self._debug('SPAN:   obj_id %d spans multiple conditions' % (query_id))
                            span_details = (query_cond, n_samples)
                            try:
                                self._results_by_condition[query_id].append( span_details )
                            except KeyError:
                                self._results_by_condition[query_id] = [ span_details ]

                            self._debug('SAVE:   %d samples (%d -> %d) for obj_id %d (%s)' % (
                                                    n_samples,
                                                    start_frame,stop_frame,
                                                    query_id,query_cond))

                            first = df.irow(0)
                            r['count'] += 1
                            r['start_obj_ids'].append( (first['x'], first['y'], query_id, first.name, start_time) )
                            r['df'].append( df )

                    this_id = _id
                    this_cond = _cond
                    csv_results = {k:[] for k in self._cols}

                elif _id == this_id:
                    #sometimes we get duplicate rows. only append if the fn is
                    #greater than the last one
                    fns = csv_results["framenumber"]
                    if (not fns) or (_framenumber > fns[-1]):
                        fns.append(_framenumber)
                        for k in self._cols:
                            if k != "framenumber":
                                csv_results[k].append(this_row[k])

                else:
                    self._warn("CANT GO BACK %d vs %d" % (_id,this_id))
                    continue
            except ValueError, e:
                self._warn("ERROR: %s\n\t%r" % (e,row))

        h5.close()



