"""
functions for building interactive command line analysis tools
"""

import os.path
import argparse
import datetime

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')
import autodata.files

from strawlab.constants import DATE_FMT

from .filters import FILTER_REMOVE, FILTER_TRIM, FILTER_NOOP

def get_default_args(**kwargs):
    """
    returns an argument parser result filled with sensible defaults.
    You probbably want to use the 
    :py:meth:`analysislib.combine.CombineH5WithCSV.add_from_uuid`
    instead
    """
    parser = get_parser()
    args = parser.parse_args("--zfilt trim --rfilt trim".split(' '))
    for k,v in kwargs.iteritems():
        setattr(args,k,v)
    return parser,args

def get_parser(*only_these_options, **defaults):
    """
    returns an ArgumentParser instance configured with common filtering
    and plotting options. This object is suitable for extending with additional
    analysis specific options, and for passing to
    :py:meth:`analysislib.combine.CombineH5WithCSV.add_from_args`
    """

    filt_choices = (FILTER_REMOVE, FILTER_TRIM, FILTER_NOOP)

    parser = argparse.ArgumentParser()
    if not only_these_options or "csv-file" in only_these_options:
        parser.add_argument(
            '--csv-file', type=str,
            help='path to *.csv file (if not using --uuid)')
    if not only_these_options or "h5-file" in only_these_options:
        parser.add_argument(
            '--h5-file', type=str,
            help='path to simple_flydra.h5 file (if not using --uuid)')
    if not only_these_options or "show-obj-ids" in only_these_options:
        parser.add_argument(
            '--show-obj-ids', action='store_true', default=False,
            help='show obj_ids on plots where appropriate')
    if not only_these_options or "reindex" in only_these_options:
        parser.add_argument(
            '--reindex', action='store_true', default=False,
            help='reindex simple_flydra h5 file')
    if not only_these_options or "show" in only_these_options:
        parser.add_argument(
            '--show', action='store_true',
            default=defaults.get('show',False),
            help='show plots')
    if not only_these_options or "cached" in only_these_options:
        parser.add_argument(
            '--cached', action='store_true',
            default=defaults.get('cached',False),
            help='load cached analysis pkl file')
    if not only_these_options or "ignore-permission-errors" in only_these_options:
        parser.add_argument(
            '--ignore-permission-errors', action='store_true',
            default=False,
            help='ignore permission errors (warning, plots might not be readable by others)')
    if not only_these_options or "no-trackingstats" in only_these_options:
        parser.add_argument(
            '--plot-tracking-stats', action='store_true', default=False,
            help='plot tracking length distribution for all flies in h5 file (takes some time)')
    if not only_these_options or "zfilt" in only_these_options:
        parser.add_argument(
            '--zfilt', type=str, choices=filt_choices,
            required=False if 'zfilt' in defaults else True,
            default=defaults.get('zfilt', 'trim'),
            help='method to filter trajectory data based on z values')
    if not only_these_options or "zfilt-min" in only_these_options:
        parser.add_argument(
            '--zfilt-min', type=float, default=0.10,
            help='minimum z, metres (default %(default)s)')
    if not only_these_options or "zfilt-max" in only_these_options:
        parser.add_argument(
            '--zfilt-max', type=float, default=0.90,
            help='maximum z, metres (default %(default)s)')
    if not only_these_options or "uuid" in only_these_options:
        parser.add_argument(
            '--uuid', type=str, nargs='*', default=None,
            help='get the appropriate csv and h5 file for this UUID (multiple may be specified)')
    if not only_these_options or "basedir" in only_these_options:
        parser.add_argument(
            '--basedir', type=str, default=None,
            help='base directory in which data files can be found by UUID'
        )
    if not only_these_options or "outdir" in only_these_options:
        parser.add_argument(
            '--outdir', type=str, default=None,
            help='directory to save plots')
    if not only_these_options or "rfilt" in only_these_options:
        parser.add_argument(
            '--rfilt', type=str, choices=filt_choices,
            required=False if 'rfilt' in defaults else True,
            default=defaults.get('rfilt', 'trim'),
            help='method to filter trajectory data based on radius from centre values')
    if not only_these_options or "rfilt-max" in only_these_options:
        parser.add_argument(
            '--rfilt-max', type=float, default=0.42,
            help='maximum r, metres, (default %(default)s)')
    if not only_these_options or "lenfilt" in only_these_options:
        parser.add_argument(
            '--lenfilt', type=float, default=1.0,
            required=False,
            help='filter trajectories shorter than this many seconds (default %(default)s)')
    if not only_these_options or "idfilt" in only_these_options:
        parser.add_argument(
            '--idfilt', type=int, default=[], nargs='*',
            help='only show these obj_ids')
    if not only_these_options or "customfilt" in only_these_options:
        parser.add_argument(
            '--customfilt', type=str, default=None,
            help='string to eval against a dataframe')
        parser.add_argument(
            '--customfilt-len', type=int, default=None,
            help='minimum length of remaining (seconds) after applying custom filter. '\
                 'note: all data is returned, it is not trimmed as per the zfilt and rfilt '\
                 'operations')
    if not only_these_options or "frames-before" in only_these_options:
        parser.add_argument(
            '--frames-before', type=int, default=0,
            help='number of frames added at the beginning of the trajectory before the trial actually starts')
    if not only_these_options or "arena" in only_these_options:
        parser.add_argument(
            '--arena', type=str, default='flycave',
            help='name of arena type')
    if not only_these_options or "tfilt" in only_these_options:
        parser.add_argument(
            '--tfilt-before', type=str,
            help='keep only trajectories before this time (%s). '\
                 'note: in local time (i.e. the times in the start_time plot)'\
                  % DATE_FMT.replace("%","%%"))
        parser.add_argument(
            '--tfilt-after', type=str,
            help='keep only trajectories after this time (%s). '\
                 'note: in local time (i.e. the times in the start_time plot)'\
                 % DATE_FMT.replace("%","%%"))

    return parser

def check_args(parser, args, max_uuids=1000):
    """
    checks that the command line arguments parsed to the parser make sense.
    In particular this checks if the number of uuids passed to idfilt
    make sense
    """
    if args.uuid:
        if None not in (args.csv_file, args.h5_file):
            parser.error("if uuid is given, --csv-file and --h5-file are not required")
        if len(args.uuid) > max_uuids:
            parser.error("only %d uuids supported" % max_uuids)
        if len(args.uuid) > 1 and args.outdir is None:
            parser.error("if multiple uuids are given, --outdir is required")
    else:
        if None in (args.csv_file, args.h5_file):
            parser.error("both --csv-file and --h5-file are required")

    for f in ("tfilt_before", "tfilt_after"):
        v = getattr(args, f, None)
        if v is not None:
            try:
                datetime.datetime.strptime(v, DATE_FMT)
            except ValueError:
                parser.error("could not parse tfilt-%s: %s" % (f,v))

