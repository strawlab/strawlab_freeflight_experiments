#!/usr/bin/env python2
import os.path
import sys
from datetime import datetime, timedelta

if not os.environ.get('DISPLAY'):
    print "DISPLAY NOT SET: USING AGG BACKEND"
    import matplotlib
    matplotlib.use('agg')

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')

from strawlab.constants import find_experiment
import autodata.files
import analysislib.filters
import analysislib.combine
import analysislib.args
import analysislib.plots as aplt
import analysislib.util as autil


def filter_trials_after_start(combine,
                              delta=timedelta(hours=2)):
    """
    Returns a new combine object with only the trials that started before exp_start + delta.

    :type combine: analysislib.combine._Combine
    """
    # retrieve the experiment start times
    datetime_limits = {}

    def keep_before(unused0, unused1, soid, uuid, unused2):
        if uuid is None:
            return False  # Can happen even in basic examples, probably because of race conditions
                          # FIXME: infer better the uuid in combine (easier if we assume only one exp at a time)
        if uuid not in datetime_limits:
            exp_md = find_experiment(uuid)[2]
            exp_start = exp_md['start_secs'] + 1E-9 * exp_md['start_nsecs']
            datetime_limits[uuid] = datetime.fromtimestamp(exp_start) + delta
        time0 = datetime.fromtimestamp(soid[-1])  # time0 is the last field
        return time0 < datetime_limits[uuid]

    return combine.filter_trials(keep_before)


if __name__=='__main__':
    parser = analysislib.args.get_parser()

    args = parser.parse_args()

    # hack: separate by uuid even if we specify an outdir, useful to debug
    if args.outdir is not None and not args.outdir.endswith(args.uuid[0]):
        args.outdir = os.path.join(args.outdir, args.uuid[0])

    analysislib.args.check_args(parser, args)

    combine = autil.get_combiner_for_args(args)
    combine.add_from_args(args)

    fname = combine.fname
    results,dt = combine.get_results()

    print "plots stored in", combine.plotdir
    print "files saved as", fname
    ncond = combine.get_num_conditions()

    # filter out trajectories that start after some time; plot
    # this is not something we want to do all the time so we can:
    #   - nasty comment-uncomment
    #   - add a new filter to the command line of combine scripts
    #   - create a script that will just do this
    #   - ...
    combine_before = filter_trials_after_start(combine, delta=timedelta(hours=6))

    aplt.save_args(combine_before, args)
    aplt.save_results(combine_before, args)

    aplt.save_most_loops(combine_before, args)

    aplt.plot_traces(combine_before, args,
                     name=fname + '.6h.traces',
                     figncols=ncond,
                     in3d=False,
                     show_starts=True,
                     show_ends=True)

    aplt.plot_traces(combine_before, args,
                     name=fname + '.6h.traces.3d',
                     figncols=ncond,
                     in3d=True)

    aplt.plot_histograms(combine_before, args,
                figncols=ncond)

    if args.plot_tracking_stats and len(args.uuid) == 1:
        fplt = autodata.files.FileView(
                  autodata.files.FileModel(show_progress=True,filepath=combine.h5_file))
        with aplt.mpl_fig("%s.tracking" % fname,args,figsize=(10,5)) as f:
            fplt.plot_tracking_data(
                        f.add_subplot(1,2,1),
                        f.add_subplot(1,2,2))

    #correlation and histogram plots
    correlations = (('stim_x','vx'),)#('stim_z','vz'))
    histograms = ("velocity","dtheta","stim_x","vx")
    correlation_options = {"stim_x:vx":{"range":[[-1,1],[-0.3,0.3]]},
                           "latencies":range(0,150,5),
                           "latencies_to_plot":(0,5,10,15,25,50,75,100,125),
    }
    histogram_options = {"normed":{"velocity":True,
                                   "dtheta":True},
                         "range":{"velocity":(0,1),
                                  "dtheta":(-20,20),
                                  "stim_x":(-2,2),
                                  "stim_y":(-2,2),
                                  "stim_z":(-2,2),
                                  "vx":(-1,1),
                                  "vy":(-1,1),
                                  "vz":(-1,1)},
                         "xlabel":{"velocity":"velocity (m/s)",
                                   "dtheta":"turn rate (rad/s)"},
    }

    if args.show:
        aplt.show_plots()

    sys.exit(0)

