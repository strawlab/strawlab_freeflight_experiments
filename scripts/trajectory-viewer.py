#!/usr/bin/env python
import sys
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
                    "uuid", "zfilt", "rfilt", "idfilt","show","reindex","lenfilt",
                    "zfilt-min","zfilt-max","rfilt-max","outdir","arena","csv-file","h5-file",
                    zfilt='none',
                    rfilt='none',
                    lenfilt=0,
    )
    parser.add_argument(
        "--animate", action="store_true")
    parser.add_argument(
        "--save", action="store_true", help="save a csv of this trajectory")
    
    args = parser.parse_args()

    analysislib.args.check_args(parser, args, max_uuids=1)

    if not args.idfilt or len(args.idfilt) != 1:
        parser.error("one obj_id must be specified")

    uuid = args.uuid[0]
    obj_id = args.idfilt[0]
    
    suffix = autil.get_csv_for_uuid(uuid)
    combine = autil.get_combiner(suffix)
    combine.calc_turn_stats = True
    combine.add_from_uuid(uuid,suffix,args=args)
    df,dt,_ = combine.get_one_result(obj_id)

    plot_axes=["theta","dtheta","rotation_rate","velocity","rcurve","ratio","radius"]
    ylimits={"omega":(-2,2),"dtheta":(-0.15,0.15),"rcurve":(0,1)}

    if args.animate:
        args.show = True
        anim = aplt.animate_infinity(
                combine, args,
                df,dt,
                plot_axes=plot_axes,
                ylimits=ylimits
        )
    else:
        aplt.plot_infinity(
                combine, args,
                df,dt,
                plot_axes=plot_axes,
                ylimits=ylimits
        )


    if args.show:
        aplt.show_plots()

    if args.save:
        df.to_csv("%s_%s.csv" % (uuid, obj_id))
        df.save("%s_%s.df" % (uuid, obj_id))

    sys.exit(0)
