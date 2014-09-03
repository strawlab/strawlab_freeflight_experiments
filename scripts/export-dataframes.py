#!/usr/bin/env python2
import sys
import os.path

import scipy.io
import numpy as np

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')

import analysislib.filters
import analysislib.args
import analysislib.plots as aplt
import analysislib.curvature as acurve
import analysislib.util as autil
import analysislib.combine as acombine

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

def _write_df(dest, df, index):
    dest = dest + '_' + aplt.get_safe_filename(index)

    kwargs = {}
    if index == 'framenumber':
        kwargs['index_label'] = 'framenumber'
    elif index.startswith('time'):
        kwargs['index_label'] = 'time'

    df.to_csv(dest+'.csv',**kwargs)
    df.to_pickle(dest+'.df')

    dict_df = df.to_dict('list')
    dict_df['index'] = df.index.values
    scipy.io.savemat(dest+'.mat', dict_df)

    print "WROTE csv,df,mat", dest

if __name__=='__main__':
    parser = analysislib.args.get_parser(
                    zfilt='none',
                    rfilt='none',
                    lenfilt=0,
    )
    parser.add_argument(
        '--n-longest', type=int, default=100,
        help='save only the N longest trajectories')
    parser.add_argument(
        '--index', default='framenumber',
        help='the index of the returned dataframe (framenumber, none, time+NN)')
    parser.add_argument(
        '--split-column',
        help='split the dataframe into two output files at the occurance of '\
             '--split-where in the given column')
    parser.add_argument(
        '--split-where', type=float, default=None,
        help='split on the first occurance of this value')

    args = parser.parse_args()

    analysislib.args.check_args(parser, args, max_uuids=1)

    combine = autil.get_combiner_for_args(args)
    combine.set_index(args.index)
    combine.add_from_args(args)

    for condition,longest in combine.get_obj_ids_sorted_by_length().iteritems():
        odir = combine.get_plot_filename(acombine.safe_condition_string(condition))
        if not os.path.isdir(odir):
            os.makedirs(odir)

        for n,(obj_id,l) in enumerate(longest):
            df,dt,(x0,y0,obj_id,framenumber0,start) = combine.get_one_result(obj_id, condition)
            dest = os.path.join(odir,'%d' % obj_id)

            if args.split_column and (args.split_where is not None):
                #find the start of the perturbation (where perturb_progress == 0)
                z = np.where(df[args.split_column].values == args.split_where)
                if len(z[0]):
                    fidx = z[0][0]
                    bdf = df.iloc[:fidx]
                    _write_df(dest+"_before", bdf, args.index)
                    adf = df.iloc[fidx:]
                    _write_df(dest+"_after", adf, args.index)
            else:
                _write_df(dest, df, args.index)

            if n >= args.n_longest:
                break

    with open(os.path.join(combine.plotdir,'README_DATA_FORMAT.txt'),'w') as f:
        f.write(FORMAT_DOCS)

    sys.exit(0)
