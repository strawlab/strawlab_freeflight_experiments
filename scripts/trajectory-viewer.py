#!/usr/bin/env python2
import sys
import os.path

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')

import analysislib.filters
import analysislib.combine
import analysislib.args
import analysislib.plots as aplt
import analysislib.curvature as acurve
import analysislib.util as autil

if __name__=='__main__':
    parser = analysislib.args.get_parser(
                    zfilt='none',
                    rfilt='none',
                    lenfilt=0,
    )
    parser.add_argument(
        '--index', default='framenumber',
        help='the index of the returned dataframe (framenumber, none, time+NN)')
    parser.add_argument(
        "--animate", action="store_true")
    parser.add_argument(
        "--save", action="store_true", help="save a csv of this trajectory")
    parser.add_argument(
        "--show-target", action="store_true", help="show target on path (useful with --animate)")
    parser.add_argument(
        "--plot-values", help="plot these fields too (comma separated list)",
        default=",".join(["theta","dtheta","rotation_rate","velocity","rcurve","ratio","radius"]))
    
    args = parser.parse_args()

    analysislib.args.check_args(parser, args, max_uuids=1)

    if not args.idfilt or len(args.idfilt) != 1:
        parser.error("one obj_id must be specified")

    uuid = args.uuid[0]
    obj_id = args.idfilt[0]

    combine = autil.get_combiner_for_args(args)
    combine.set_index(args.index)
    combine.add_from_args(args)

    plot_axes = args.plot_values.split(',')

    ylimits={"omega":(-2,2),"dtheta":(-20,20),"rcurve":(0,1)}

    results, dt = combine.get_results()

    anims = []
    for i,(current_condition,r) in enumerate(results.iteritems()):
        for df,(x0,y0,_obj_id,framenumber0,time0) in zip(r['df'], r['start_obj_ids']):
            if _obj_id == obj_id:

                name = analysislib.combine.safe_condition_string(current_condition)
                title = "%s: %s" % (obj_id, current_condition)

                if args.animate:
                    args.show = True
                    anim = aplt.animate_infinity(
                            combine, args,
                            df,dt,
                            plot_axes=plot_axes,
                            ylimits=ylimits,
                            title=title,
                            show_trg=args.show_target and ('trg_x' in df.columns)
                    )
                    anims.append(anim)
                else:
                    aplt.plot_infinity(
                            combine, args,
                            df,dt,
                            name=name,
                            plot_axes=plot_axes,
                            ylimits=ylimits,
                            title=title,
                    )

                if args.save:
                    basedir = args.outdir if args.outdir else combine.plotdir
                    df.to_csv(os.path.join(basedir,"%s_%s_%s.csv" % (uuid, obj_id, name)))
                    df.save(os.path.join(basedir,"%s_%s_%s.df" % (uuid, obj_id, name)))

    if args.show:
        aplt.show_plots()

    sys.exit(0)
