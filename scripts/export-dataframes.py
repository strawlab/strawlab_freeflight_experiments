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

    sys.exit(0)
