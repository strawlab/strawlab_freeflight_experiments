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
import analysislib.fixes
import analysislib.features
import analysislib.plots as aplt
import analysislib.curvature as acurve
import analysislib.util as autil

if __name__=='__main__':
    parser = analysislib.args.get_parser(disable_filters=True)

    parser.add_argument(
        '--index', default='framenumber',
        help='the index of the returned dataframe (framenumber, none, time+NN)')
    parser.add_argument(
        "--animate", action="store_true")
    parser.add_argument(
        "--no-h264", action="store_true",
        help="don't save h264 format movie (because your ffmpeg is too old)")
    parser.add_argument(
        "--save-animation", action="store_true", help="save an mp4 of this trajectory")
    parser.add_argument(
        "--save-plot", action="store_true", help="save this plot to the plot_dir")
    parser.add_argument(
        "--show-target", action="store_true", help="show target on path (useful with --animate)")
    parser.add_argument(
        "--show-starts", action="store_true")
    parser.add_argument(
        "--plot-values", help="plot these features too", nargs='+',
        default=("theta","dtheta","rotation_rate","velocity","ratio","radius"))
    parser.add_argument(
        "--test-filter-args",
        help="test filter args (e.g. '--xfilt triminterval --yfilt triminterval --vfilt triminterval --zfilt trim', or '--arena flycave')")
    parser.add_argument(
        "--ylimits", help="y-axis limits name:min:max,[name2:min2:max2]")
    parser.add_argument(
        "--no-disable-filters", action='store_true',
        help='do not disable the normal arena based filters')
    parser.add_argument(
        "--help-features", action='store_true',
        help="list all the features that can be plotted")

    args = parser.parse_args()
    if args.help_features:
        print "Available features (may not include some already in the csv)"
        for c in analysislib.features.get_all_columns(True):
            print "  * %s" % c
        parser.exit(0)

    analysislib.args.check_args(parser, args, max_uuids=1)

    try:
        uuid = args.uuid[0]
    except TypeError:
        uuid = '0'*32
    obj_ids = map(int,args.idfilt)
    plot_axes = args.plot_values

    combine = autil.get_combiner_for_args(args)
    combine.set_index(args.index)
    for p in plot_axes:
        try:
            combine.add_feature(column_name=p)
        except ValueError, e:
            print e.message + " ... assuming it is in the csv"

    combine.add_from_args(args)

    if args.save_animation:
        args.animate = True

    if args.outdir:
        basedir = args.outdir
    elif args.save_animation:
        basedir = strawlab.constants.get_movie_dir(uuid)
        if not os.path.isdir(basedir):
            os.makedirs(basedir)
    else:
        basedir = combine.plotdir

    results, dt = combine.get_results()

    if args.test_filter_args:
        filt_parser = analysislib.args.get_parser(arena=args.arena)
        filt_args = filt_parser.parse_args(args.test_filter_args.split(' '))
        if args.animate:
            parser.error('--animate and --test-filter-args can not be combined')
    else:
        filt_args = None

    cmdline_limits = {}
    if args.ylimits:
        for field in args.ylimits.split(','):
            n,m,M = field.split(":")
            cmdline_limits[n] = (float(m),float(M))

    anims = {}
    for i,(current_condition,r) in enumerate(results.iteritems()):
        for df,(x0,y0,obj_id,framenumber0,time0) in zip(r['df'], r['start_obj_ids']):
            if obj_id in obj_ids:

                rrate_max_abs = analysislib.fixes.get_rotation_rate_limit_for_plotting(combine)

                ylimits = {"dtheta":(-20,20),"rcurve":(0,1), "rotation_rate":(-rrate_max_abs,rrate_max_abs)}
                ylimits.update(cmdline_limits)

                name = analysislib.combine.safe_condition_string(current_condition)


                if args.save_plot:
                    # full path
                    filename = combine.get_plot_filename("%s_%s_%s" % (obj_id, framenumber0, name))
                elif args.save_animation:
                    # this gets joined to a full path later
                    filename = "%s_%s_%s" % (obj_id, framenumber0, name)
                else:
                    filename = '/tmp/trajectory_viewer'

                title = "%s: %s (fn0 %d)" % (obj_id, combine.get_condition_name(current_condition), framenumber0)

                if args.animate:
                    args.show = True
                    anim = aplt.animate_infinity(
                            combine, args,
                            df,dt,
                            ylimits=ylimits,
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
                            ylimits=ylimits,
                            name=filename,
                            plot_axes=plot_axes,
                            title=title,
                            show_filter_args=filt_args,
                            show_starts_ends=args.show_starts
                    )

    if args.save_animation:
        Writer = animation.writers['ffmpeg']
        for anim in anims:
            filename,uuid,obj_id,current_condition = anims[anim]
            title = "%s: obj_id %s (condition %s)" % (uuid, obj_id, current_condition)
            filename = os.path.join(basedir,filename + ".mp4")
            print "WRITING MP4:",filename
            if args.no_h264:
                writer = Writer(fps=15, metadata=dict(title=title), bitrate=1800)
                anim.save(filename, writer=writer)
                print "WROTE mp4v", filename
            else:
                try:
                    writer = Writer(fps=15, metadata=dict(title=title), extra_args=['-vcodec', 'h264', '-pix_fmt', 'yuv420p'])
                    anim.save(filename, writer=writer)
                    print "WROTE h264", filename
                except RuntimeError:
                    print "FAILED TO WRITE h264: Your FFMPEG is too old. Upgrade or call with --no-h264"

    if (not args.save_animation) and args.show:
        aplt.show_plots()

    sys.exit(0)
