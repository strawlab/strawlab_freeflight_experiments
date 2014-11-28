import os.path
import sys
import random
import time
import cPickle
import pickle
import re
import operator
import hashlib
import datetime
import calendar
from collections import defaultdict

import tables
import pandas as pd
import numpy as np
import pytz
import scipy.io

import roslib

roslib.load_manifest('strawlab_freeflight_experiments')

import autodata.files

import analysislib.fixes
import analysislib.filters
import analysislib.args
import analysislib.curvature as acurve

from ros_flydra.constants import IMPOSSIBLE_OBJ_ID, IMPOSSIBLE_OBJ_ID_ZERO_POSE
from strawlab.constants import DATE_FMT, AUTO_DATA_MNT

from whatami import What, MAX_EXT4_FN_LENGTH

#results = {
#   condition:{
#       df:[dataframe,...],
#       start_obj_ids:[(x0,y0,obj_id,framenumber0,time0),...],
#       count:n_frames
#   }
#}

def safe_condition_string(s):
    return "".join([c for c in s if re.match(r'\w', c)])

class _Combine(object):

    calc_linear_stats = True
    calc_angular_stats = True
    calc_turn_stats = True

    def __init__(self, **kwargs):
        self._enable_debug = kwargs.get("debug",True)
        self._enable_warn = kwargs.get("warn",True)
        self._dt = None
        self._lenfilt = None
        self._idfilt = []
        self._skipped = {}
        self._results = {}
        self._uuids = defaultdict(list)
        self._custom_filter = None
        self._custom_filter_min = None
        self._tzname = 'Europe/Vienna'
        self._tfilt_before = None
        self._tfilt_after = None
        self._tfilt = None
        self._plotdir = None
        self._analysistype = None
        self._index = 'framenumber'
        self._warn_cache = {}

        self._configdict = {'v':6,  #bump this version when you change delicate combine machinery
                            'index':self._index
        }

    def set_index(self, index):
        VALID_INDEXES = ('framenumber','none')
        if (index not in VALID_INDEXES) and (not index.startswith('time')):
            raise ValueError('index must be one of %s,time+NN (where NN is a pandas resample specifier)' % ', '.join(VALID_INDEXES))
        self._index = index
        self._configdict['index'] = self._index

    def configuration(self):
        return What(self.__class__.__name__, self._configdict)

    def _args_to_configuration(self, args):
        for k,v in args._get_kwargs():
            if k not in analysislib.args.DATA_MODIFYING_ARGS:
                continue
            if k in ('uuid','idfilt') and v is not None and len(v):
                self._configdict[k] = sorted(v)
            else:
                self._configdict[k] = v

    def _get_cache_name_and_config_string(self):
        s = self.configuration().as_string()
        if len(s) > (MAX_EXT4_FN_LENGTH - 4): #4=.pkl
            fn = hashlib.sha224(s).hexdigest() + '.pkl'
        else:
            fn = s + '.pkl'
        return os.path.join(AUTO_DATA_MNT,'cached','combine',fn), s

    def _get_cache_name(self):
        return self._get_cache_name_and_config_string()[0]

    def _get_cache_file(self):
        pkl = self._get_cache_name()
        if os.path.exists(pkl):
            self._debug("IO:     reading %s" % pkl)
            with open(pkl,"r+b") as f:
                # Speed optimisation, see:
                #   http://stackoverflow.com/questions/16833124/pickle-faster-than-cpickle-with-numeric-data
                # and
                #   http://stackoverflow.com/questions/19807790/
                #   given-a-pickle-dump-in-python-how-to-i-determine-the-used-protocol
                def unpickle_fast():
                    import pickletools
                    with open(pkl, 'r') as reader:
                        if next(pickletools.genops(reader))[0].proto >= 2:
                            return cPickle.load(f)
                        return pickle.load(f)
                return unpickle_fast()
        return None

    def _save_cache_file(self):
        pkl,s = self._get_cache_name_and_config_string()
        WRITE_PKL=bool(int(os.environ.get('WRITE_PKL','1')))
        with open(pkl,"w+b") as f:
            self._debug("IO:     writing %s" % pkl)
            cPickle.dump({"results":self._results,
                          'uuids': self._uuids,
                          "dt":self._dt,
                          "csv_file":self.csv_file},
                         f, protocol=pickle.HIGHEST_PROTOCOL)

        #if the string has been truncted to a hash then also write a text file with
        #the calibration string
        if os.path.basename(os.path.splitext(pkl)[0]) != s:
            with open(pkl.replace('.pkl','.txt'),"w") as f:
                f.write(s)

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
        if self._enable_debug:
            print m

    def _warn(self, m):
        if self._enable_warn:
            print m

    def _warn_once(self, m):
        if m not in self._warn_cache:
            self._warn(m)
            self._warn_cache[m] = 0
        self._warn_cache[m] += 1

    @property
    def analysis_type(self):
        if self._analysistype is None:
            return os.path.basename(sys.argv[0])
        else:
            return self._analysistype

    @analysis_type.setter
    def analysis_type(self, val):
        self._analysistype = val

    @property
    def plotdir(self):
        """the directory in which to store plots for this analysis"""
        if self._plotdir is None:
            #none before a csv file has been added / args have been processed
            #(because of --outdir)
            return self._plotdir

        pd = os.path.join(self._plotdir, self.analysis_type)
 
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

    def get_plot_filename(self, name, subdir=''):
        """return a full path to the autodata directory to save any plots"""
        if subdir:
            pd = os.path.join(self.plotdir,subdir)
            if not os.path.isdir(pd):
                os.makedirs(pd)
        return os.path.join(self.plotdir,subdir,name)

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
        self._enable_debug = True

    def disable_debug(self):
        self._enable_debug = False

    def enable_warn(self):
        self._enable_warn = True

    def disable_warn(self):
        self._enable_warn = False

    def get_results(self):
        """Returns all data for all conditions that passed the configured filters

        Returns:
            A two-tuple.

            the first retured variable is a nested dict of the following 
            structure

            results = {
               condition:{
                   df:[dataframe,...],
                   start_obj_ids:[(x0,y0,obj_id,framenumber0,time0),...],
                   count:n_frames
               }

            The second is dt

        """
        return self._results, self._dt

    def get_uuids(self):
        """Returns the uuid for each trial in results.

        Returns a dictionary {condition: [uuid,...]}.

        Historical note:
          This was added after get_results, where it would belong.
          Adding uuids allow to uniquelly identify a trajectory, playing well
          with metadata retrieval, the flydata package and generally completing
          the information provided by combine, while not breaking the many dependents
          of the get_results method.
        """
        return self._uuids

    def get_one_result(self, obj_id, condition=None):
        """
        Get the data associated with a single object_id

        returns: (dataframe, dt, (x0,y0,obj_id,framenumber0,time0))
        """
        if (not condition) and (obj_id in self.get_spanned_results()):
            raise ValueError("obj_id: %s exists in multiple conditions - please specify which one" % obj_id)

        for i,(current_condition,r) in enumerate(self._results.iteritems()):
            for df,(x0,y0,_obj_id,framenumber0,time0) in zip(r['df'], r['start_obj_ids']):
                if _obj_id == obj_id:
                    if (not condition) or (current_condition == condition):
                        return df,self._dt,(x0,y0,obj_id,framenumber0,time0)

        raise ValueError("No such obj_id: %s (in condition: %s)" % (obj_id, condition))

    def get_obj_ids_sorted_by_length(self):
        """
        Get a sorted list of the longest trajectories
        returns: a dictionary condition:[(obj_id,len),...]
        """
        best = {}
        for i,(current_condition,r) in enumerate(self._results.iteritems()):
            if not r['count']:
                continue
            for df,(x0,y0,obj_id,framenumber0,time0) in zip(r['df'], r['start_obj_ids']):
                try:
                    best[current_condition][obj_id] = len(df)
                except KeyError:
                    best[current_condition] = {obj_id:len(df)}

        sorted_best = {c:sorted(best[c].items(), key=operator.itemgetter(1), reverse=True) for c in self.get_conditions()}
        return sorted_best

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
        self._ninfinity = kwargs.get('ninfinity', 5)

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
                    df = self.get_fake_infinity(self._ninfinity,0,0,0,framenumber,0,0,self._dt)
                else:
                    df = self.get_fake_infinity(
                                n_infinity=self._ninfinity,
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
                self._results[cond].append('FAKEUUID')

                obj_id += 1
                framenumber += len(df)

    @staticmethod
    def get_fake_infinity_trajectory(n_infinity, random_stddev, x_offset, y_offset, frame0, npts=200):
        def get_noise(d):
            if random_stddev:
                return (np.random.random(len(d)) - 0.5) * random_stddev
            else:
                return np.zeros_like(d)

        pi = np.pi
        leaft = np.linspace(-pi/4.,pi/4., npts//2)
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
    def get_fake_infinity(n_infinity, random_stddev, x_offset, y_offset, frame0, nan_pct, latency, dt, npts=200):
        df = _CombineFakeInfinity.get_fake_infinity_trajectory(n_infinity, random_stddev, x_offset, y_offset, frame0, npts)

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
        self._args_to_configuration(args)

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
        self._args_to_configuration(args)

        if not csv_suffix:
            csv_suffix = self._csv_suffix

        self._idfilt = args.idfilt
        self._maybe_add_tfilt(args)

        if args.uuid:
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

        # uuid
        assert 'exp_uuid' in df.columns, 'exp_uuid not in CSV, cannot infer the UUID'

        df = df.fillna(method="pad")
        df['time'] = df['t_sec'] + (df['t_nsec'] * 1e-9)

        dt = (df['time'].values[-1] - df['time'].values[0]) / len(df)

        if self._dt is None:
            self._dt = dt
            self._lenfilt = lenfilt
        else:
            #check the new csv file was recorded with the same timebase
            assert abs(dt-self._dt) < 1e-4

        for cond,dfc in df.groupby('condition'):  # SANTI FIXME: here use block grouping

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

                assert dfo['exp_uuid'].nunique() == 1, 'cond=%s, oid=%d has no unique UUID !?!?' % (cond, obj_id)
                uuid = dfo['exp_uuid'].unique()[0]

                self._results[cond]['df'].append(dfo)
                self._results[cond]['start_obj_ids'].append(self._get_result(dfo))
                self._results[cond]['count'] += 1
                self._uuids[cond].append(uuid)


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
        self._args_to_configuration(args)

        self._maybe_add_tfilt(args)
        self._maybe_add_customfilt(args)

        if args.uuid:
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
        # N.B. uuid is also stored in the h5file, in case we ever need it

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
        #framenumber must be present
        cols = ["framenumber","tnsec","tsec"]
        cols.extend(csv_cols)
        #some rows are handled differently
        #condition, exp_uuid, flydra_data_file are strings,
        #lock_object, t_sec, t_nsec have to be present
        self._cols = set(cols) - set(['condition','lock_object','t_sec','t_nsec','exp_uuid','flydra_data_file'])

        self._csv_suffix = kwargs.get("csv_suffix")

        #use this for keeping track of results that span multiple conditions
        self._results_by_condition = {}


    def add_from_uuid(self, uuid, csv_suffix=None, **kwargs):
        """Add a csv and h5 file collected from the experiment with the
        given uuid
        """
        if 'args' in kwargs:
            args = kwargs['args']
        else:
            parser,args = analysislib.args.get_default_args()
            for k in kwargs:
                setattr(args,k,kwargs[k])

        args.uuid = [uuid]
        self.add_from_args(args, csv_suffix=csv_suffix)

    def add_from_args(self, args, csv_suffix=None):
        """Add possibly multiple csv and h5 files based on the command line
        arguments given
        """
        self._args_to_configuration(args)
        if args.cached:
            d = self._get_cache_file()
            if d is not None:
                self._results = d['results']
                self._uuids = d['uuids']
                self._dt = d['dt']
                self.csv_file = d['csv_file']   #for plot names

                if len(args.uuid) > 1:
                    self.plotdir = args.outdir
                else:
                    fm = autodata.files.FileModel(basedir=args.basedir)
                    fm.select_uuid(args.uuid[0])
                    self.plotdir = args.outdir if args.outdir else fm.get_plot_dir()

                return

        if not csv_suffix:
            csv_suffix = self._csv_suffix

        self._maybe_add_tfilt(args)
        self._maybe_add_customfilt(args)

        if args.uuid:
            if len(args.uuid) > 1:
                self.plotdir = args.outdir

            for uuid in args.uuid:
                fm = autodata.files.FileModel(basedir=args.basedir)
                fm.select_uuid(uuid)
                try:
                    csv_file = fm.get_file_model(csv_suffix).fullpath
                    h5_file = fm.get_file_model("simple_flydra.h5").fullpath
                except autodata.files.NoFile:
                    if len(args.uuid) == 1:
                        raise
                    else:
                        continue

                #this handles the single uuid case
                if self.plotdir is None:
                    self.plotdir = args.outdir if args.outdir else fm.get_plot_dir()

                self.add_csv_and_h5_file(csv_file, h5_file, args, uuid=uuid)

        else:
            csv_file = args.csv_file
            h5_file = args.h5_file

            self.plotdir = args.outdir if args.outdir else os.getcwd()

            self.add_csv_and_h5_file(csv_file, h5_file, args)

        self._save_cache_file()

    def get_spanned_results(self):
        spanned = {}
        for oid,details in self._results_by_condition.iteritems():
            if len(details) > 1:
                spanned[oid] = details
        return spanned

    def add_csv_and_h5_file(self, csv_fname, h5_file, args, uuid=None):
        """Add a single csv and h5 file"""

        self.csv_file = csv_fname
        self.h5_file = h5_file

        fix = analysislib.fixes.load_fixups(csv_file=self.csv_file,
                                            h5_file=self.h5_file)

        self._debug("IO:     reading %s" % csv_fname)
        self._debug("IO:     reading %s" % h5_file)
        if fix.active:
            self._debug("IO:     fixing data %s" % fix)

        #open the csv file as a dataframe
        try:
            csv = pd.read_csv(self.csv_file,na_values=('None',),
                              error_bad_lines=False,
                              dtype={'framenumber':int,
                                     'condition':str,
                                     'exp_uuid':str,
                                     'flydra_data_file':str})
        except:
            self._warn("ERROR: possibly corrupt csv. Re-parsing %s" % self.csv_file)
            #protect against rubbish in the framenumber column
            csv = pd.read_csv(self.csv_file,na_values=('None',),
                              error_bad_lines=False,
                              low_memory=False,
                              dtype={'framenumber':float,
                                     'condition':str,
                                     'exp_uuid':str,
                                     'flydra_data_file':str})
            csv = csv.dropna(subset=['framenumber'])
            csv['framenumber'] = csv['framenumber'].astype(int)

        # uuid from csv
        assert 'exp_uuid' in csv.columns, 'exp_uuid not in CSV, cannot infer the UUID'

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

        #minimum length of 2 to prevent later errors calculating derivitives
        dur_samples = max(2,self.min_num_frames)

        results = self._results
        skipped = self._skipped

        for (oid,cond),odf in csv.groupby(('lock_object','condition')):

            assert odf['exp_uuid'].nunique() == 1, 'cond=%s, oid=%d has no unique UUID !?!?' % (cond, oid)
            uuid = odf['exp_uuid'].unique()[0]

            if oid in (IMPOSSIBLE_OBJ_ID,IMPOSSIBLE_OBJ_ID_ZERO_POSE):
                continue

            if args.idfilt and (oid not in args.idfilt):
                continue

            if fix.active:
                cond = fix.fix_condition(cond)

            if cond not in results:
                results[cond] = dict(count=0,
                                      start_obj_ids=[],
                                      df=[])
                skipped[cond] = 0

            r = results[cond]

            #the csv may be written at a faster rate than the framerate,
            #causing there to be multiple rows with the same framenumber.
            #find the last index for all unique framenumbers for this trial
            fdf = odf.drop_duplicates(cols=('framenumber',),take_last=True)
            trial_framenumbers = fdf['framenumber'].values

            #get the comparible range of data from flydra
            if args.frames_before != 0:
                start_frame = trial_framenumbers[0] - args.frames_before
            else:
                start_frame = trial_framenumbers[0]

            query = "(obj_id == %d) & (framenumber >= %d) & (framenumber < %d)" % (
                            oid,
                            start_frame,
                            trial_framenumbers[-1])

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

            validframenumber = valid['framenumber'][valid_cond]

            n_samples = len(validframenumber)
            if n_samples < dur_samples:
                self._debug('TRIM:   %d samples for obj_id %d' % (n_samples,oid))
                self._skipped[cond] += 1
                continue

            traj_start_frame = validframenumber[0]
            traj_stop_frame = validframenumber[-1]
            traj_start = h5.root.trajectory_start_times.readWhere("obj_id == %d" % oid)

            flydra_series = []
            for a in 'xyz':
                avalid = valid[a][valid_cond]
                flydra_series.append( pd.Series(avalid,name=a,index=validframenumber) )

            #we can now create a dataframe that has the flydra data, and the
            #original index of the csv dataframe
            framenumber_series = pd.Series(validframenumber,name='framenumber',index=validframenumber)
            flydra_series.append(framenumber_series)

            #make a ns since epoch column
            tns0 = (traj_start['first_timestamp_secs'] * 1e9) + traj_start['first_timestamp_nsecs']
            tns = ((validframenumber - traj_start_frame) * self._dt * 1e9) + tns0
            tns_series = pd.Series(tns,name='tns',index=validframenumber,dtype=np.uint64)
            flydra_series.append(tns_series)

            df = pd.concat(flydra_series,axis=1)

            try:
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
                        self._debug('FILTER: %d for obj_id %d' % (n_samples,oid))
                        self._skipped[cond] += 1
                        df = None
            except Exception, e:
                self._skipped[cond] += 1
                self._warn("ERROR: could not calc trajectory metrics for oid %s (%s long)\n\t%s" % (oid,n_samples,e))
                continue

            start_time = float(csv.head(1)['t_sec'] + (csv.head(1)['t_nsec'] * 1e-9))
            if not self._maybe_apply_tfilt_should_save(start_time):
                df = None

            if df is not None:

                n_samples = len(df)
                span_details = (cond, n_samples)
                try:
                    self._results_by_condition[oid].append( span_details )
                except KeyError:
                    self._results_by_condition[oid] = [ span_details ]

                self._debug('SAVE:   %d samples (%d -> %d) for obj_id %d (%s)' % (
                                        n_samples,
                                        traj_start_frame,traj_stop_frame,
                                        oid,cond))

                if self._index == 'framenumber':
                    #if the csv has been written at a faster rate than the
                    #flydra data then fdf contains the last estimate in the
                    #csv for that framenumber (because drop_duplicates take_last=True)
                    #removes the extra rows and make a new framenumber index
                    #unique.
                    #
                    #an outer join allows the tracking data to have started
                    #before the csv (frames_before)

                    #delete the framenumber from the h5 dataframe, it only
                    #duplicates what should be in the index anyway
                    del df['framenumber'] #df.drop('framenumber', axis=1, inplace=True) (drop added in 13.1)

                    #if there are any columns common in both dataframes the result
                    #seems to be that the concat resizes the contained values
                    #by adding an extra dimenstion.
                    #df['x'].values.ndim = 1 becomes = 2 (for version of
                    #pandas < 0.14). To work around this, remove any columns
                    #in the csv dataframe that exists in df
                    common_columns = df.columns & fdf.columns
                    for c in common_columns:
                        self._warn_once('ERROR: renaming duplicated colum name "%s" to "_%s"' % (c,c))
                        cv = df[c].values
                        del df[c]
                        df['_'+c] = cv

                    df = pd.concat((
                                fdf.set_index('framenumber'),df),
                                axis=1,join='outer')

                    #restore a framenumber column for API compatibility
                    df['framenumber'] = df.index.values

                    if df['x'].values.ndim > 1:
                        self._warn_once("ERROR: pandas merge added empty dimension to dataframe values")

                    # Because of the outer join, trim filter do not work
                    # (trimmed observations come back as haunting missing values)
                    # This is a quick workaround...
                    df = df.dropna(subset=['x'])
                    # TODO: check for holes

                elif (self._index == 'none') or (self._index.startswith('time')):

                    if self._index.startswith('time'):
                        #add a tns column
                        odf['tns'] = np.array((odf['t_sec'].values * 1e9) + odf['t_nsec'], dtype=np.uint64)

                    #we still must trim the original dataframe by the trim conditions (framenumber)
                    odf_fns = odf['framenumber'].values
                    odf_fn0_idx = np.where(odf_fns >= traj_start_frame)[0][0]   #first frame
                    odf_fnN_idx = np.where(odf_fns <= traj_stop_frame)[0][-1]   #last frame

                    #in this case we want to keep all the rows (outer)
                    #but the two dataframes should remain sorted by
                    #framenumber because we use that for building a new time index
                    #if we resample
                    df = pd.merge(
                                odf.iloc[odf_fn0_idx:odf_fnN_idx],df,           #trim as filtered
                                suffixes=("_csv","_h5"),
                                on='framenumber',
                                left_index=False,right_index=False,
                                how='outer',sort=True)

                    #in the time case we want to set a datetime index and optionally resample
                    if self._index.startswith('time'):
                        try:
                            _,resamplespec = self._index.split('+')
                        except ValueError:
                            resamplespec = None

                        if df['framenumber'][0] != traj_start_frame:
                            dfv = df['framenumber'].values
                            #now the df is sorted we can just remove the invalid data from the front
                            n_invalid_rows = np.where(dfv==traj_start_frame)[0][0]

                            self._warn("ERROR: csv started %s rows before tracking (fn csv:%r... vs h5:%s, obj_id) %s" % (n_invalid_rows,dfv[0:3],traj_start_frame,oid))

                            df = df.iloc[n_invalid_rows:]

                        df['tns'] = ((df['framenumber'].values - traj_start_frame) * self._dt * 1e9) + tns0

                        df['datetime'] = df['tns'].values.astype('datetime64[ns]')
                        #any invalid (NaT) rows break resampling
                        df = df.dropna(subset=['datetime'])
                        df = df.set_index('datetime')

                        if resamplespec is not None:
                            df = df.resample(resamplespec, fill_method='pad')

                else:
                    raise Exception('Unknown index requested %s' % self._index)

                if fix.should_fix_rows:
                    for _ix, row in df.iterrows():
                        fixed = fix.fix_row(row)
                        for col in fix.should_fix_rows:
                            if col not in df.columns:
                                self._warn_once("ERROR: column '%s' missing from dataframe (are you resampling?)" % col)
                                continue
                            #modify in place
                            try:
                                df.loc[_ix,col] = fixed[col]
                            except IndexError, e:
                                self._warn("ERROR: could not apply fixup to obj_id %s (column '%s')" % (oid,col))

                #the start time and the start framenumber are defined by the experiment,
                #so they come from the csv (fdf)
                first = fdf.irow(0)

                start_time = float(first['t_sec'] + (first['t_nsec'] * 1e-9))
                start_framenumber = int(first['framenumber'])
                #i could get this from the merged dataframe, but this is easier...
                #also, the >= is needed to make valid['x'][0] not crash
                #because for some reason sometimes we have a framenumber
                #in the csv (which means it must be tracked) but not the simple
                #flydra file....?
                #
                #maybe there is an off-by-one hiding elsewhere
                query = "(obj_id == %d) & (framenumber >= %d)" % (oid, start_framenumber)
                valid = trajectories.readWhere(query)
                start_x = valid['x'][0]
                start_y = valid['y'][0]

                r['count'] += 1
                r['start_obj_ids'].append( (start_x, start_y, oid, start_framenumber, start_time) )
                r['df'].append( df )
                self._uuids.append(uuid)

        h5.close()


FORMAT_DOCS = """
Exported Data Formats
=====================

Introduction
------------

Files are exported in three different formats csv,pandas dataframe (df)
and matlab (mat).

Depending on the choice of index and the output format the final data should
be interpreted with the following knowledge.

General Concepts
----------------
All data is a combination of that collected from the tracking system
(at precisely 1/frame_rate intervals) and that recorded by the experiment
(at any interval). Frame rate is typically 100 or 120Hz.

The tracking data includes
 * x,y,z (position, m)
 * framenumber
 * tns (time in nanoseconds)
 * vx,vy,vz,velocity (velocity, m/s)
 * ax,ay,az (acceleration, m/s2)
 * theta (heading, rad)
 * dtheta (turn rate, rad/s)
 * radius (distance from origin, m)
 * omega (?)
 * rcurve (radius of curvature of current turn, m)

The experimental data contained in the csv file can include any other fields,
however it is guarenteed to contain at least the following
 * t_sec (unix time seconds component)
 * t_nsec (unix time, sub-second component as nanoseconds)
 * framenumber
 * condition (string)
 * lock_object
 * exp_uuid (string)

** Note **
When the tracking data and the experimental data is combined, any columns that
are identically named in both will be renamed. Tracking data colums that have been
renamed are suffixed with '_h5' while csv columns are added '_csv'

Index Choice
------------
The index choice of the files is denoted by the filename suffix; _framenumber,
_none, _time. According to the choice of index the two sources of data (tracking
and csv) are combined as follows.

** Framenumber index **
The most recent (in time) record for each framenumber is taken from the
experimental (csv) data. If the csv was written at a faster rate than the
tracking data some records will be lost. If the csv was written slower
than the tracking data then there will be missing elements in the columns
of data originating from the csv.

Data with a framenumber index will not contain framenumber column(s) but will
contain tns and t_sec/nsec columns

The framenumber is guarenteed to only increase, but may do so non-monotonically
(due to missing tracking data for example)

** No (none) index **
All records from tracking and experimental data are combined together (temporal
order is preserved). Columns may be missing elements. The wall-clock time for the
tracking data is stored in tns. The wall-clock time for experimental csv
data rows can be reconstructed from t_sec+t_nsec.

Data with no index will contain framenumber columns (with _csv and _h5 suffixes)
and will also contain tns and t_sec/nsec columns

** Time index **
Data with time index is often additionally resampled, indicated by the
file name being timeXXX where X is an integer. If resampled, the string XXX
is defined here -
http://pandas.pydata.org/pandas-docs/dev/timeseries.html#offset-aliases

For example a file named _time10L has been resampled to a 10 millisecond timebase.

This is time aware resampling, so any record from either source that did not
occur at the time instant is resampled. Data is up-sampled by padding the
most recent value forward, and down-sampled by taking the mean over the
interval.

Data with a time index will contain framenumber columns (with _csv and _h5 suffixes)
and tns and t_sec/nsec columns.

If the data has NOT been resampled the data may still contain missing rows

Output Format
-------------
In addition to the colum naming and data combining overview just given,
the following things should be considered when loading exported data. 

** csv files **
The first colum contains the index. If 'framenumber' was chosen the column is
labeled 'framenumber'. If 'none' index was chosen the column is
left unlabeled and the values are monotonically increasing integers. If 'time'
was chosen the column is labeled 'time' and contains strings of the
format '%Y-%m-%d_%H:%M:%S.%f'

** mat files **
The mat files will also contain a variable 'index' which is an integer
for 'framenumber' and 'none' types. If If the index type is 'time' then the values
are nanoseconds since unix epoch.

** df files **
Pandas dataframes should contain information as previously described and also
the data types and values for all with full precision

"""

def write_result_dataframe(dest, df, index, to_df=True, to_csv=True, to_mat=True):
    dest = dest + '_' + safe_condition_string(index)

    kwargs = {}
    if index == 'framenumber':
        kwargs['index_label'] = 'framenumber'
    elif index.startswith('time'):
        kwargs['index_label'] = 'time'

    if to_csv:
        df.to_csv(dest+'.csv',**kwargs)

    if to_df:
        df.to_pickle(dest+'.df')

    if to_mat:
        dict_df = df.to_dict('list')
        dict_df['index'] = df.index.values
        scipy.io.savemat(dest+'.mat', dict_df)

    formats = ('csv' if to_csv else '',
               'df' if to_df else '',
               'mat' if to_mat else '')

    return "%s.{%s}" % (dest, ','.join(filter(len,formats)))

