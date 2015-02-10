#!/usr/bin/env python2
import sys
import os.path

if not os.environ.get('DISPLAY'):
    print "DISPLAY NOT SET: USING AGG BACKEND"
    import matplotlib
    matplotlib.use('agg')

import matplotlib.animation as animation

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')
import strawlab.constants

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
                    trajectory_start_offset=0.0
    )
    parser.add_argument(
        '--index', default='framenumber',
        help='the index of the returned dataframe (framenumber, none, time+NN)')
    parser.add_argument(
        "--animate", action="store_true")
    parser.add_argument(
        "--save-data", action="store_true", help="save a csv of this trajectory")
    parser.add_argument(
        "--save-animation", action="store_true", help="save an mp4 of this trajectory")
    parser.add_argument(
        "--show-target", action="store_true", help="show target on path (useful with --animate)")
    parser.add_argument(
        "--plot-values", help="plot these fields too (comma separated list)",
        default=",".join(["theta","dtheta","rotation_rate","velocity","rcurve","ratio","radius"]))
    parser.add_argument(
        "--test-filter-args")
    
    args = parser.parse_args()

    analysislib.args.check_args(parser, args, max_uuids=1)

    uuid = args.uuid[0]
    obj_ids = map(int,args.idfilt)

    combine = autil.get_combiner_for_args(args)
    combine.set_index(args.index)
    combine.add_from_args(args)

    if args.outdir:
        basedir = args.outdir
    elif args.save_animation:
        basedir = strawlab.constants.get_movie_dir(uuid, camera=combine.analysis_type)
        if not os.path.isdir(basedir):
            os.makedirs(basedir)
    else:
        basedir = combine.plotdir

    plot_axes = args.plot_values.split(',')

    results, dt = combine.get_results()

    if args.test_filter_args:
        filt_parser = analysislib.args.get_parser(arena=args.arena)
        filt_args = filt_parser.parse_args(args.test_filter_args.split(' '))
    else:
        filt_args = None

    anims = {}
    for i,(current_condition,r) in enumerate(results.iteritems()):
        for df,(x0,y0,obj_id,framenumber0,time0) in zip(r['df'], r['start_obj_ids']):
            if obj_id in obj_ids:

                name = analysislib.combine.safe_condition_string(current_condition)

                filename = "%s_%s_%s" % (uuid, obj_id, name)
                title = "%s: %s" % (obj_id, combine.get_condition_name(current_condition))

                if args.animate:
                    args.show = True
                    anim = aplt.animate_infinity(
                            combine, args,
                            df,dt,
                            plot_axes=plot_axes,
                            title=title,
                            show_trg=args.show_target and ('trg_x' in df.columns),
                            repeat=not args.save_animation
                    )
                    anims[anim] = (filename, uuid, obj_id, current_condition)
                else:
                    aplt.plot_infinity(
                            combine, args,
                            df,dt,
                            name=name,
                            plot_axes=plot_axes,
                            title=title,
                            show_filter_args=filt_args
                    )

                if args.save_data:
                    df.to_csv(os.path.join(basedir,filename + ".csv"))
                    df.save(os.path.join(basedir,filename + ".df"))

    if args.save_animation:
        Writer = animation.writers['ffmpeg']
        for anim in anims:
            filename,uuid,obj_id,current_condition = anims[anim]
            title = "%s: obj_id %s (condition %s)" % (uuid, obj_id, current_condition)
            filename = os.path.join(basedir,filename + ".mp4")
            print "WRITING MP4:",filename
            try:
                writer = Writer(fps=15, metadata=dict(title=title), extra_args=['-vcodec', 'h264', '-pix_fmt', 'yuv420p'])
                anim.save(filename, writer=writer)
                print "WROTE h264"
            except RuntimeError:
                writer = Writer(fps=15, metadata=dict(title=title), bitrate=1800)
                anim.save(filename, writer=writer)
                print "WROTE mp4v"

    if (not args.save_animation) and args.show:
        aplt.show_plots()

    sys.exit(0)
