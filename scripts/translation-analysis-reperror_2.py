import os.path
import sys

if not os.environ.get('DISPLAY'):
    print "DISPLAY NOT SET: USING AGG BACKEND"
    import matplotlib
    matplotlib.use('agg')

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')

import autodata.files
import analysislib.filters
import analysislib.combine
import analysislib.args
import analysislib.plots as aplt
import analysislib.curvature as curve
import analysislib.util as autil

import analysislib
print '#'*100
print analysislib.__file__
print '#'*100


if __name__=='__main__':
    parser = analysislib.args.get_parser()

    args = parser.parse_args()

    analysislib.args.check_args(parser, args)

    combine = autil.get_combiner_for_args(args)
    combine.set_features()
    combine.add_feature(column_name='mean_reproj_error_px')
    combine.add_feature(column_name='visible_in_n_cams')
    combine.add_feature(column_name='err_pos_stddev_m')
    combine.add_feature(column_name='radius')
    combine.add_feature(column_name='velocity')
    combine.add_feature(column_name='dtheta')
    combine.add_feature(column_name='theta')
    combine.add_feature(column_name='ax')
    combine.add_feature(column_name='ay')
    combine.add_feature(column_name='az')
    combine.add_feature(column_name='vx')
    combine.add_feature(column_name='vy')
    combine.add_feature(column_name='vz')
    combine.add_from_args(args)
    fname = combine.fname
    results,dt = combine.get_results()

    sys.exit(0)

