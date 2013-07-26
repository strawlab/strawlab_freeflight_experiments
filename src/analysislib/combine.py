import os.path
import sys
import argparse
import Queue
import random
import time

import tables
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import roslib

roslib.load_manifest('flycave')
import autodata.files
import analysislib.filters
import analysislib.combine
import analysislib.args
import analysislib.curvature
import analysislib.plots as aplt

from ros_flydra.constants import IMPOSSIBLE_OBJ_ID, IMPOSSIBLE_OBJ_ID_ZERO_POSE

#results = {
#   condition:{
#       df:[dataframe,...],
#       start_obj_ids:[(x0,y0,obj_id,framenumber0,time0),...]
#       count:[n_frames,...],
#   }
#}

class _Combine(object):

    plotdir = None

    def __init__(self, **kwargs):
        self._debug = kwargs.get("debug",True)
        self._dt = None
        self._lenfilt = None
        self._skipped = {}
        self._results = {}

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
                self.warn("obj_id column not indexed, this will be slow. reindex")

        return trajectories

    @property
    def fname(self):
        return os.path.join(self.plotdir,os.path.basename(self.csv_file).split('.')[0])

    @property
    def min_num_frames(self):
        try:
            return self._lenfilt / self._dt
        except TypeError:
            return 1

    @property
    def framerate(self):
        return 1.0 / self._dt

    def get_num_skipped(self, condition):
        return self._skipped.get(condition,0)

    def get_num_analysed(self, condition):
        return self._results[condition]['count']

    def get_num_trials(self, condition):
        return self.get_num_skipped(condition) + self.get_num_analysed(condition)

    def get_total_trials(self):
        return sum([self.get_num_trials(c) for c in self.get_conditions()])

    def get_conditions(self):
        return self._results.keys()

    def get_num_frames(self, seconds):
        return seconds / self._dt

    def enable_debug(self):
        self._debug = True

    def disable_debug(self):
        self._debug = False

    def debug(self, m):
        if self._debug:
            print m

    def warn(self, m):
        print m

    def get_results(self):
        return self._results, self._dt

    def get_one_result(self, obj_id):
        for i,(current_condition,r) in enumerate(self._results.iteritems()):
            for df,(x0,y0,_obj_id,framenumber0,time0) in zip(r['df'], r['start_obj_ids']):
                if _obj_id == obj_id:
                    return df,self._dt,(x0,y0,obj_id,framenumber0,time0)

    def get_result_columns(self):
        for current_condition,r in self._results.iteritems():
            for df in r['df']:
                return list(df.columns)
        return []

    def add_custom_filter(self, s, post_filter_min):
        if 'df[' not in s:
            raise Exception("incorrectly formatted filter string: %s" % s)
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

        obj_id = 1
        framenumber = 1

        for c in range(self._nconditions):
            cond = "tex%d/svg/1.0/1.0/adv/..." % c

            try:
                self._results[cond]
            except KeyError:
                self._results[cond] = {"df":[],"start_obj_ids":[],"count":0}

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
        cols.extend( analysislib.curvature.calc_velocities(df, dt) )
        cols.extend( analysislib.curvature.calc_angular_velocities(df, dt) )

        #add some uncorrelated noise to rrate
        rrate = (df['dtheta'].values * 10.0) + (0.0 * (np.random.random(len(df)) - 0.5))

        if latency > 0:
            rrate = np.concatenate( (rrate[latency:],np.random.random(latency)) )

        #rotation rate comes from a csv file, with a lower freq, so trim some % of the data
        #to simulate missing values
        trim = np.random.random(len(rrate)) < (nan_pct / 100.0)
        rrate[trim] = np.nan

        df['rotation_rate'] = rrate
        df['v_offset_rate'] = np.zeros_like(df['az'].values)

        for c in cols:
            del df[c]

        return df

    def add_from_args(self, args):
        self.plotdir = (args.outdir if args.outdir else os.getcwd()) + "/"
        self.csv_file = "test"

class CombineCSV(_Combine):

    csv_file = ''

    def __init__(self, **kwargs):
        _Combine.__init__(self, **kwargs)

    def add_from_args(self, args):
        self.plotdir = (args.outdir if args.outdir else os.getcwd()) + "/"
        self.add_csv_file(args.csv_file, args.lenfilt)

    def add_csv_file(self, csv_file, lenfilt=None):
        self._lenfilt = lenfilt

        self.debug("reading %s" % csv_file)
        self.csv_file = csv_file
        df = pd.DataFrame.from_csv(self.csv_file,index_col="framenumber")

        assert 'lock_object' in df.columns

        self._df = df.fillna(method="pad")
        self._df['time'] = self._df['t_sec'] + (self._df['t_nsec'] * 1e-9)

        self._dt = (self._df['time'].values[-1] - self._df['time'].values[0]) / len(self._df)

        results = {}

        for cond,dfc in self._df.groupby('condition'):
            results[cond] = {'df':[],'start_obj_ids':[],'count':0}
            for obj_id,dfo in dfc.groupby('lock_object'):
                if obj_id == 0:
                    continue

                if not self._df_ok(dfo):
                    continue

                results[cond]['df'].append(dfo)
                results[cond]['start_obj_ids'].append(self._get_result(dfo))
                results[cond]['count'] += 1

        self._results = results

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
        if args.uuid is not None:
            uuid = args.uuid[0]
            fm = autodata.files.FileModel(basedir=args.basedir)
            fm.select_uuid(uuid)
            h5_file = fm.get_file_model("simple_flydra.h5").fullpath
        else:
            h5_file = args.h5_file

        self.add_h5_file(h5_file)

    def add_from_uuid(self, uuid, *args, **kwargs):
        fm = autodata.files.FileModel(basedir=os.environ.get("FLYDRA_AUTODATA_BASEDIR"))
        fm.select_uuid(uuid)
        h5_file = fm.get_file_model("simple_flydra.h5").fullpath
        self.add_h5_file(h5_file)

    def add_h5_file(self, h5_file):
        self.debug("reading %s" % h5_file)

        warnings = {}

        self.h5_file = h5_file

        h5 = tables.openFile(h5_file, mode='r')

        self._trajectories = self._get_trajectories(h5)
        dt = 1.0/self._trajectories.attrs['frames_per_second']

        self._trajectory_start_times = h5.root.trajectory_start_times

        if self._dt is None:
            self._dt = dt
        else:
            assert dt == self._dt

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

        return df,self._dt,(traj['x'][0],traj['y'][0],obj_id,traj['framenumber'][0],t0)

class CombineH5WithCSV(_Combine):

    csv_file = ''
    h5_file = ''

    def __init__(self, loggerklass, *csv_rows, **kwargs):
        _Combine.__init__(self, **kwargs)
        self._lklass = loggerklass
        rows = ["framenumber"]
        rows.extend(csv_rows)
        self._rows = set(rows)

    def add_from_uuid(self, uuid, csv_suffix, plotdir=None, frames_before=0, **kwargs):
        fm = autodata.files.FileModel(basedir=os.environ.get("FLYDRA_AUTODATA_BASEDIR"))
        fm.select_uuid(uuid)
        csv_file = fm.get_file_model(csv_suffix).fullpath
        h5_file = fm.get_file_model("simple_flydra.h5").fullpath

        self.plotdir = (plotdir if plotdir else os.getcwd()) + "/"

        args = analysislib.args.get_default_args()
        for k in kwargs:
            setattr(args,k,kwargs[k])

        self.add_csv_and_h5_file(csv_file, h5_file, args, frames_before)

    def add_from_args(self, args, csv_suffix, frames_before=0):
        if args.uuid is not None:
            if len(args.uuid) > 1:
                self.plotdir = args.outdir + "/"

            for uuid in args.uuid:
                fm = autodata.files.FileModel(basedir=args.basedir)
                fm.select_uuid(uuid)
                csv_file = fm.get_file_model(csv_suffix).fullpath
                h5_file = fm.get_file_model("simple_flydra.h5").fullpath

                #this handles the single uuid case
                if self.plotdir is None:
                    self.plotdir = (args.outdir if args.outdir else fm.get_plot_dir()) + "/"

                self.add_csv_and_h5_file(csv_file, h5_file, args, frames_before)

        else:
            csv_file = args.csv_file
            h5_file = args.h5_file

            self.plotdir = (args.outdir if args.outdir else os.getcwd()) + "/"

            self.add_csv_and_h5_file(csv_file, h5_file, args, frames_before)

    def add_csv_and_h5_file(self, csv_fname, h5_file, args, frames_before=0):
        self.debug("reading %s" % csv_fname)
        self.debug("reading %s" % h5_file)

        warnings = {}

        self.csv_file = csv_fname
        self.h5_file = h5_file

        infile = self._lklass(fname=csv_fname, mode="r")

        h5 = tables.openFile(h5_file, mode='r+' if args.reindex else 'r')

        trajectories = self._get_trajectories(h5)
        dt = 1.0/trajectories.attrs['frames_per_second']

        if self._dt is None:
            self._dt = dt
            self._lenfilt = args.lenfilt
        else:
            assert dt == self._dt

        dur_samples = self.min_num_frames

        _ids = Queue.Queue(maxsize=2)
        this_id = IMPOSSIBLE_OBJ_ID
        csv_results = {}

        results = self._results
        this_row = {}

        skipped = self._skipped

        for row in infile.record_iterator():
            try:

                _cond = str(row.condition)
                _id = int(row.lock_object)
                _t = float(row.t_sec) + (float(row.t_nsec) * 1e-9)
                _framenumber = int(row.framenumber)

                for k in self._rows:
                    if k != "framenumber":
                        try:
                            this_row[k] = float(getattr(row,k))
                        except AttributeError:
                            this_row[k] = np.nan
                            if k not in warnings:
                                self.warn("WARNING: no such column in csv:%s" % k)
                                warnings[k] = True

                if not _cond in results:
                    results[_cond] = dict(count=0,
                                          start_obj_ids=[],
                                          df=[])
                    skipped[_cond] = 0
                    

                if _id == IMPOSSIBLE_OBJ_ID_ZERO_POSE:
                    continue
                if _id == IMPOSSIBLE_OBJ_ID:
                    continue
                elif _id != this_id:
                    try:
                        query_id,query_framenumber,start_time,query_cond = _ids.get(False)
                    except Queue.Empty:
                        #first time
                        this_id = _id
                        csv_results = {k:[] for k in self._rows}
                        query_id = None
                    finally:
                        _ids.put((_id,_framenumber,_t,_cond),block=False)

                    #first time
                    if query_id is None:
                        continue

                    if (not args.idfilt) or (query_id in args.idfilt):

                        r = results[query_cond]

                        if frames_before < 0:
                            query = "obj_id == %d" % query_id
                        else:
                            query = "(obj_id == %d) & (framenumber >= %d)" % (
                                        query_id,
                                        query_framenumber-frames_before)

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

                        if n_samples < dur_samples: # must be at least this long
                            self.debug('insufficient samples (%d) for obj_id %d' % (n_samples,query_id))
                            self._skipped[_cond] += 1
                        else:
                            self.debug('%s %d: frame0 %d, %d samples'%(_cond, query_id,
                                                                valid[0]['framenumber'],
                                                                n_samples))

                            dfd = {'x':validx,'y':validy,'z':validz}

                            for k in self._rows:
                                if k != "framenumber":
                                    dfd[k] = pd.Series(csv_results[k],index=csv_results['framenumber'])

                            df = pd.DataFrame(dfd,index=validframenumber)

                            r['count'] += 1
                            r['start_obj_ids'].append(  (validx[0], validy[0], query_id, query_framenumber, start_time) )
                            r['df'].append( df )

                    this_id = _id
                    csv_results = {k:[] for k in self._rows}

                elif _id == this_id:
                    #sometimes we get duplicate rows. only append if the fn is
                    #greater than the last one
                    fns = csv_results["framenumber"]
                    if (not fns) or (_framenumber > fns[-1]):
                        fns.append(_framenumber)

                        for k in self._rows:
                            if k != "framenumber":
                                csv_results[k].append(this_row[k])

                else:
                    self.warn("CANT GO BACK %d vs %d" % (_id,this_id))
                    continue
            except ValueError, e:
                self.warn("ERROR: %s\n\t%r" % (e,row))

        h5.close()



