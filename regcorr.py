import itertools

import numpy as np
import pandas as pd

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')
import analysislib.args as aargs
import analysislib.curvature as acurve
import analysislib.util as autil
import analysislib.plots as aplt

from analysislib.plots import LEGEND_TEXT_BIG, LEGEND_TEXT_SML

parser,args = aargs.get_default_args(
            uuid=["2834d4f0bcab11e2ad5d6c626d3a008a"],
            outdir='/tmp/',
            show=True,
            zfilt_max=0.85,
)

combine = autil.get_combiner("rotation.csv")
combine.add_from_args(args, "rotation.csv")

correlations = ['rotation_rate','dtheta']
correlation_options = {i:{} for i in correlations}

flat_data,nens = acurve.flatten_data(args, combine, correlations)

RESOLUTION = 0.1
regress = ['rotation_rate','dtheta']

xaxis_mids = np.arange(0,1.5,RESOLUTION) + RESOLUTION/2.0
yaxis_slope = {k:[] for k in nens}
yaxis_r2value = {k:[] for k in nens}

for cond in nens:
    data = {k:np.concatenate(flat_data[k][cond]) for k in regress}
    rr,dt = acurve.remove_nans(*list(data[k] for k in regress))

    #in steps of RESOLUTION regress rotation rate against dtheta
    for i in np.arange(0,1.5,RESOLUTION):
        arr = np.abs(rr)
        valid = (arr > i) & (arr < i+RESOLUTION)
        
        _rr = rr[valid]
        _dt = dt[valid]

        slope, intercept, r_value, p_value, std_err = acurve.calculate_linregress(_rr, _dt)

        yaxis_slope[cond].append(slope)
        yaxis_r2value[cond].append(r_value**2)

for name,yaxis in [("slope",yaxis_slope),("r2",yaxis_r2value)]:
    with aplt.mpl_fig("%s_regress" % name, args) as fig:
            ax = fig.gca()
            for current_condition in sorted(nens):
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
