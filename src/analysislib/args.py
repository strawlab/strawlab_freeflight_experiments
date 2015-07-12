"""
functions for building interactive command line analysis tools
"""

import os.path
import argparse
import datetime
import numpy as np

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')
import autodata.files

from strawlab.constants import DATE_FMT

from .filters import FILTER_REMOVE, FILTER_TRIM, FILTER_NOOP, FILTER_TRIM_INTERVAL, FILTER_TYPES
from .arenas import get_arena_from_args

def _filter_types_args():
    a = []
    for f in FILTER_TYPES:
        name = "%sfilt" % f
        a.extend((name, name+"_min", name+"_max", name+"_interval"))
    return a

REQUIRED_ARENA_DEFAULTS = ["trajectory_start_offset"]

DATA_MODIFYING_ARGS = [
    'idfilt',
    'uuid',
    'arena',
    'lenfilt',
]
DATA_MODIFYING_ARGS.extend(_filter_types_args())
DATA_MODIFYING_ARGS.extend(REQUIRED_ARENA_DEFAULTS)

class _ArenaAwareArgumentParser(argparse.ArgumentParser):
    def parse_args(self, *args, **kwargs):
        args = argparse.ArgumentParser.parse_args(self, *args, **kwargs)

        try:
            arena = get_arena_from_args(args)
        except ValueError, e:
            self.error(e.message)

        #disable-filters can override
        if (not getattr(args,"no_disable_filters",False)) and getattr(args,"disable_filters",False):
            for f in arena.filters:
                f.disable()

            args.lenfilt = 0
            args.trajectory_start_offset = 0.0

        #for forensics we store all configuration on the args object
        for f in arena.filters:
            f.set_on_args(args)

        defaults = arena.get_filter_defaults()
        for p in REQUIRED_ARENA_DEFAULTS:
            if getattr(args,p,None) is None:
                setattr(args,p,defaults[p])

        return args

def get_default_args(**kwargs):
    """
    returns an argument parser result filled with sensible defaults.
    You probbably want to use the 
    :py:meth:`analysislib.combine.CombineH5WithCSV.add_from_uuid`
    instead
    """
    if 'arena' not in kwargs:
        kwargs['arena'] = 'flycave'
    parser = get_parser(**kwargs)
    args = parser.parse_args('')
    return parser,args

def get_parser(*only_these_options, **defaults):
    """
    returns an ArgumentParser instance configured with common filtering
    and plotting options. This object is suitable for extending with additional
    analysis specific options, and for passing to
    :py:meth:`analysislib.combine.CombineH5WithCSV.add_from_args`
    """

    filt_choices = (FILTER_REMOVE, FILTER_TRIM, FILTER_NOOP, FILTER_TRIM_INTERVAL)

    parser = _ArenaAwareArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    if not only_these_options or "csv-file" in only_these_options:
        parser.add_argument(
            '--csv-file', type=str,
            default=defaults.get('csv_file',None),
            help='path to *.csv file (if not using --uuid)')
    if not only_these_options or "h5-file" in only_these_options:
        parser.add_argument(
            '--h5-file', type=str,
            default=defaults.get('h5_file',None),
            help='path to simple_flydra.h5 file (if not using --uuid)')
    if not only_these_options or "show-obj-ids" in only_these_options:
        parser.add_argument(
            '--show-obj-ids', action='store_true',
            default=defaults.get('show_obj_ids', False),
            help='show obj_ids on plots where appropriate')
    if not only_these_options or "reindex" in only_these_options:
        parser.add_argument(
            '--no-reindex', action='store_false', dest='reindex',
            default=defaults.get('no_reindex') if 'no_reindex' in defaults else not defaults.get('reindex', False),
            help='reindex simple_flydra h5 file')
    if not only_these_options or "show" in only_these_options:
        parser.add_argument(
            '--show', action='store_true',
            default=defaults.get('show',False),
            help='show plots')
    if not only_these_options or "cached" in only_these_options:
        parser.add_argument(
            '--no-cached', action='store_false', dest='cached',
            default=not defaults.get('no_cached') if 'no_cached' in defaults else not defaults.get('cached',False),
            help='load cached analysis pkl file')
    if not only_these_options or "recache" in only_these_options:
        parser.add_argument(
            '--recache', action='store_true', dest='recache',
            default=defaults.get('recache') if 'recached' in defaults else defaults.get('recache', False),
            help='force to recache even if the cache already exists')
    if not only_these_options or "ignore-permission-errors" in only_these_options:
        parser.add_argument(
            '--ignore-permission-errors', action='store_true',
            default=defaults.get('ignore_permission_errors', False),
            help='ignore permission errors (warning, plots might not be readable by others)')
    if not only_these_options or "no-trackingstats" in only_these_options:
        parser.add_argument(
            '--plot-tracking-stats', action='store_true',
            default=defaults.get('no_trackingstats', False),
            help='plot tracking length distribution for all flies in h5 file (takes some time)')

    if not only_these_options or "disable-filters" in only_these_options:
        parser.add_argument(
            '--disable-filters', action='store_true',
            default=defaults.get('disable_filters',False),
            help='disables all filters (overrides other command line options)')
    for i,desc in FILTER_TYPES.iteritems():
        if not only_these_options or ("%sfilt" % i) in only_these_options:
            parser.add_argument(
                '--%sfilt' % i, type=str, choices=filt_choices,
                default=defaults.get('%sfilt' % i, None),
                help='method to filter trajectory data based on %s values' % i)
        if not only_these_options or ("%sfilt-min" % i) in only_these_options:
            parser.add_argument(
                '--%sfilt-min' % i, type=float,
                default=defaults.get('%sfilt_min' % i, None),
                help='minimum %s' % desc)
        if not only_these_options or ("%sfilt-max" % i) in only_these_options:
            parser.add_argument(
                '--%sfilt-max' % i, type=float,
                default=defaults.get('%sfilt_max' % i, None),
                help='maximum %s' % desc)
        if not only_these_options or ("%sfilt-interval" % i) in only_these_options:
            parser.add_argument(
                '--%sfilt-interval' % i, type=float,
                default=defaults.get('%sfilt_interval' % i, None),
                help="when using 'triminterval' filter methods, the number of seconds over "\
                     "which the filter must match in order for data to be trimmed. ")

    if not only_these_options or "uuid" in only_these_options:
        ud = defaults.get('uuid', None)
        if ud is not None:
            ud = ud if isinstance(ud,list) else [ud]
        parser.add_argument(
            '--uuid', type=str, nargs='*',
            default=ud,
            help='get the appropriate csv and h5 file for this UUID (multiple may be specified)')
    if not only_these_options or "basedir" in only_these_options:
        parser.add_argument(
            '--basedir', type=str,
            default=defaults.get('basedir', None),
            help='base directory in which data files can be found by UUID'
        )
    if not only_these_options or "outdir" in only_these_options:
        parser.add_argument(
            '--outdir', type=str,
            default=defaults.get('outdir', None),
            help='directory to save plots')
    if not only_these_options or "lenfilt" in only_these_options:
        parser.add_argument(
            '--lenfilt', type=float,
            default=defaults.get('lenfilt', 1.0),
            help='filter trajectories shorter than this many seconds')
    if not only_these_options or "idfilt" in only_these_options:
        parser.add_argument(
            '--idfilt', type=int, nargs='*',
            default=defaults.get('idfilt', []),
            help='only show these obj_ids')
    if not only_these_options or "trajectory-start-offset" in only_these_options:
        parser.add_argument(
            '--trajectory-start-offset', type=float,
            default=defaults.get('trajectory_start_offset',None),
            help='number of seconds to relative to the start of each trial '\
                 '(i.e. from the csv) trajectory from which to keep data. if negative '\
                 'this means include data before the trial began. if positive this '\
                 'ignores data at the start of a trajectory')
    if not only_these_options or "arena" in only_these_options:
        parser.add_argument(
            '--arena', type=str,
            default=defaults.get('arena', 'flycave'),
            required=True if 'arena' not in defaults else False,
            help='name of arena type')
    if not only_these_options or "tfilt" in only_these_options:
        parser.add_argument(
            '--tfilt-before', type=str,
            default=defaults.get('tfilt_before', None),
            help='keep only trajectories before this time (%s). '\
                 'note: in local time (i.e. the times in the start_time plot)'\
                  % DATE_FMT.replace("%","%%"))
        parser.add_argument(
            '--tfilt-after', type=str,
            default=defaults.get('tfilt_after', None),
            help='keep only trajectories after this time (%s). '\
                 'note: in local time (i.e. the times in the start_time plot)'\
                 % DATE_FMT.replace("%","%%"))
    if not only_these_options or "check" in only_these_options:
        parser.add_argument(
            '--check', action='store_true',
            default=defaults.get('check', False),  # if defaults is True, then action='store_true' flips? (see other flags)
            required=False,
            help='enable dynamic checks on data invariants (i.e. check for bad smells in combined data)')

    return parser

def check_args(parser, args, max_uuids=1000, defaults_from_arena=True):
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
    else:
        if None in (args.csv_file, args.h5_file):
            parser.error("either --uuid or both --csv-file and --h5-file are required")

    od = getattr(args,'outdir',None)
    if od is not None:
        if not os.path.isdir(od):
            os.makedirs(od)

    for f in ("tfilt_before", "tfilt_after"):
        v = getattr(args, f, None)
        if v is not None:
            try:
                datetime.datetime.strptime(v, DATE_FMT)
            except ValueError:
                parser.error("could not parse tfilt-%s: %s" % (f,v))

