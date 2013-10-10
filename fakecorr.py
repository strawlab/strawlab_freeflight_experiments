#!/usr/bin/env python
import os.path
import unittest
import tempfile
import itertools
import numpy as np

import roslib
import roslib.packages
roslib.load_manifest('strawlab_freeflight_experiments')
import analysislib.combine
import analysislib.util as autil
import analysislib.args as aargs
import analysislib.plots as aplt
import analysislib.curvature as acurve
import strawlab.constants

from analysislib.plots import LEGEND_TEXT_BIG, LEGEND_TEXT_SML

FAKE_DATA = False

tdir = tempfile.mkdtemp()
uuid = '0'*32
obj_id = 5
ddir = os.path.join(
            roslib.packages.get_pkg_dir('strawlab_freeflight_experiments'),
           'test','data'
)
#make autodata look in the tempdir for file
os.environ['FLYDRA_AUTODATA_BASEDIR'] = ddir
os.environ['FLYDRA_AUTODATA_PLOTDIR'] = tdir

combine = autil.get_combiner("rotation.csv")
combine.calc_turn_stats = True
parser,args = analysislib.args.get_default_args(
            uuid=[uuid],
            outdir=tdir
)
combine.add_from_args(args)

#correlation and histogram plots
correlations = (('rotation_rate','dtheta'),)
histograms = ("velocity","dtheta","rcurve")
correlation_options = {i[0]:{} for i in correlations}
histogram_options = {"normed":{"velocity":True,
                               "dtheta":True,
                               "rcurve":True},
                     "range":{"velocity":(0,1),
                              "dtheta":(-0.5,0.5),
                              "rcurve":(0,1)},
                     "xlabel":{"velocity":"velocity (m/s)",
                               "dtheta":"turn rate (rad/s)",
                               "rcurve":"radius of curvature (m)"},
}
flatten_columns = set(list(itertools.chain.from_iterable(correlations)) + list(histograms))

flat_data,nens = acurve.flatten_data(args, combine, flatten_columns)

max_corr_at_latency = acurve.plot_correlation_analysis(args, combine, flat_data, nens, correlations, correlation_options)

regression = ('rotation_rate','dtheta')

#acurve.plot_regression_analysis(args, combine, flat_data, nens, regression, regression_options={}, resolution=0.1)

RESOLUTION = 0.1

xaxis_mids = np.arange(0,1.5,RESOLUTION) + RESOLUTION/2.0
yaxis_slope = {k:[] for k in nens}
yaxis_r2value = {k:[] for k in nens}

for cond in nens:
    data = {k:np.concatenate(flat_data[k][cond]) for k in regression}
    rr,dt = acurve.remove_nans(*list(data[k] for k in regression))

    shift,cmax = max_corr_at_latency[cond]
    if cmax < 0.05:
        continue

    if shift == 0:
        shift_rr = rr
        shift_dt = dt
    else:
        shift_rr = rr[0:-shift]
        shift_dt = dt[shift:]

    #in steps of RESOLUTION regress rotation rate against dtheta
    for i in np.arange(0,1.5,RESOLUTION):
        arr = shift_rr#np.abs(shift_rr)
        valid = (arr > i) & (arr < i+RESOLUTION)
        
        _rr = shift_rr[valid]
        _dt = shift_dt[valid]

        slope, intercept, r_value, p_value, std_err = acurve.calculate_linregress(_rr, _dt)

        yaxis_slope[cond].append(slope)
        yaxis_r2value[cond].append(r_value**2)

for name,yaxis in [("slope",yaxis_slope),("r2",yaxis_r2value)]:
    with aplt.mpl_fig("%s_regress" % name, args) as fig:
        ax = fig.gca()
        for current_condition in sorted(nens):

            if current_condition in ("gray.png/infinity.svg/0.3/-10.0/0.1/0.20", "checkerboard16.png/infinity.svg/0.0/-10.0/0.1/0.00"):
                continue

            ax.plot(
                xaxis_mids,
                yaxis[current_condition],
                marker='+',
                label=current_condition)

        ax.legend(loc='upper right', numpoints=1,
            prop={'size':LEGEND_TEXT_BIG} if len(nens) <= 4 else {'size':LEGEND_TEXT_SML},
        )
        ax.set_xlabel("rotation rate (rad/s)")
        ax.set_ylabel(name)

aplt.show_plots()

