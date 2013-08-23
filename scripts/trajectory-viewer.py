#!/usr/bin/env python
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
                    "zfilt-min","zfilt-max","rfilt-max","outdir","arena",
                    zfilt='none',
                    rfilt='none',
                    lenfilt=0,
                    show=True
    )
    parser.add_argument(
        "--animate", action="store_true")
    
    args = parser.parse_args()

    if not args.uuid or len(args.uuid) != 1:
        parser.error("one uuid must be specified")
    if not args.idfilt or len(args.idfilt) != 1:
        parser.error("one obj_id must be specified")

    uuid = args.uuid[0]
    
    suffix = autil.get_csv_for_uuid(uuid)
    combine = autil.get_combiner(suffix)
    combine.calc_turn_stats = True
    combine.add_from_uuid(uuid,suffix,args=args)
    df,dt,_ = combine.get_one_result(args.idfilt[0])

    plot_axes=["theta","dtheta","rotation_rate","velocity","rcurve","ratio","radius"]
    ylimits={"omega":(-2,2),"dtheta":(-0.15,0.15),"rcurve":(0,1)}

    if args.animate:
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

            
    aplt.show_plots()

