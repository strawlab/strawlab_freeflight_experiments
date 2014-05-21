#!/usr/bin/env python
import sys
import os.path

import scipy.io

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')

import analysislib.filters
import analysislib.args
import analysislib.plots as aplt
import analysislib.curvature as acurve
import analysislib.util as autil
import analysislib.combine as acombine

def _write_df(dest, df, drop, fillna):
    if drop:
        #remove empty cols then missing rows
        _df = df.dropna(axis=1,how='all').dropna(axis=0,how='any')
        dest += "_drop"
    elif fillna:
        _df = df.fillna(method=fillna)
        dest += "_fillna"
    else:
        _df = df

    _df.to_csv(dest+'.csv')
    scipy.io.savemat(dest+'.mat', _df.to_dict('list'))

    print dest

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
        '--drop', type=int, default=0,
        help='drop rows containing more than this many NaNs')
    parser.add_argument(
        '--fillna', type=str, choices=['ffill','bfill'],
        help='fill missing values')


    args = parser.parse_args()

    analysislib.args.check_args(parser, args, max_uuids=1)

    uuid = args.uuid[0]
    combine = autil.get_combiner_for_uuid(uuid)
    combine.add_from_uuid(uuid,args=args)

    for condition,longest in combine.get_obj_ids_sorted_by_length().iteritems():
        odir = combine.get_plot_filename(acombine.safe_condition_string(condition))
        if not os.path.isdir(odir):
            os.makedirs(odir)

        for n,(obj_id,l) in enumerate(longest):
            df,dt,(x0,y0,obj_id,framenumber0,start) = combine.get_one_result(obj_id, condition)
            dest = os.path.join(odir,'%d' % obj_id)

            _write_df(dest, df, args.drop, args.fillna)

            if n >= args.n_longest:
                break

    sys.exit(0)
