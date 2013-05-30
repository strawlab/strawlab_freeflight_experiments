import os.path
import sys
import argparse
import Queue
import pandas

import tables
import pandas
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import roslib

roslib.load_manifest('flycave')
import autodata.files
import analysislib.filters
import analysislib.combine
import analysislib.args
import analysislib.plots as aplt

from ros_flydra.constants import IMPOSSIBLE_OBJ_ID, IMPOSSIBLE_OBJ_ID_ZERO_POSE

class CombineH5WithCSV(object):

    plotdir = None
    csv_file = ''
    h5_file = ''

    def __init__(self, loggerklass, *csv_rows):
        self._lklass = loggerklass

        rows = ["framenumber"]
        rows.extend(csv_rows)
        self._rows = set(rows)

        self._results = {}
        self._skipped = {}
        self._dt = None
        self._lenfilt = None

    @property
    def fname(self):
        return os.path.join(self.plotdir,os.path.basename(self.csv_file).split('.')[0])

    @property
    def min_num_frames(self):
        return self._lenfilt / self._dt

    @property
    def framerate(self):
        return 1.0 / self._dt

    def get_num_skipped(self, condition):
        return self._skipped.get(condition,0)

    def get_num_analysed(self, condition):
        return self._results[condition]['count']

    def get_num_trials(self, condition):
        return self.get_num_skipped(condition) + self.get_num_analysed(condition) 

    def get_num_frames(self, seconds):
        return seconds / self._dt

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
        print "reading", csv_fname
        print "reading", h5_file

        warnings = {}

        self.csv_file = csv_fname
        self.h5_file = h5_file

        infile = self._lklass(fname=csv_fname, mode="r")

        h5 = tables.openFile(h5_file, mode='r+' if args.reindex else 'r')
        trajectories = h5.root.trajectories

        #unexplainable protip - adding an index on the framenumber table makes
        #things slloooowwww
        if trajectories.cols.framenumber.is_indexed:
            trajectories.cols.framenumber.removeIndex()

        if not trajectories.cols.obj_id.is_indexed:
            try:
                trajectories.cols.obj_id.createIndex()
            except tables.exceptions.FileModeError:
                print "obj_id column not indexed, this will be slow. reindex"

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
                                print "WARNING: no such column in csv:",k
                                warnings[k] = True

                if not _cond in results:
                    results[_cond] = dict(count=0,
                                          framenumber=[],
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
                            print 'insufficient samples (%d) for obj_id %d' % (n_samples,query_id)
                            self._skipped[_cond] += 1
                        else:
                            print '%s %d: frame0 %d, %d samples'%(_cond, query_id,
                                                                valid[0]['framenumber'],
                                                                n_samples)

                            dfd = {'x':validx,'y':validy,'z':validz}

                            for k in self._rows:
                                if k != "framenumber":
                                    dfd[k] = pandas.Series(csv_results[k],index=csv_results['framenumber'])

                            df = pandas.DataFrame(dfd,index=validframenumber)

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
                    print "CANT GO BACK %d vs %d" % (_id,this_id)
                    continue
            except ValueError, e:
                print "ERROR: ", e
                print row

        h5.close()

    def get_results(self):
        return self._results, self._dt

