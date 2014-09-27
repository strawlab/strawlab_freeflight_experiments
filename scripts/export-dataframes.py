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
                acombine.write_result_dataframe(dest, df, args.index)
                #find the start of the perturbation (where perturb_progress == 0)
                z = np.where(df[args.split_column].values == args.split_where)
                if len(z[0]):
                    fidx = z[0][0]
                    bdf = df.iloc[:fidx]
                    acombine.write_result_dataframe(dest+"_before", bdf, args.index)
                    adf = df.iloc[fidx:]
                    acombine.write_result_dataframe(dest+"_after", adf, args.index)
            else:
                acombine.write_result_dataframe(dest, df, args.index)

            if n >= args.n_longest:
                break

    with open(os.path.join(combine.plotdir,'README_DATA_FORMAT.txt'),'w') as f:
        f.write(acombine.FORMAT_DOCS)

    sys.exit(0)
