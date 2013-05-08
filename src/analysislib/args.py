import os.path
import argparse

import roslib
roslib.load_manifest('flycave')
import autodata.files

from .filters import FILTER_REMOVE, FILTER_TRIM, FILTER_NOOP

def get_default_args():
    parser = get_parser()
    return parser.parse_args("--zfilt trim --rfilt trim".split(' '))

def get_parser(*only_these_options):
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
            '--show', action='store_true', default=False,
            help='show plots')
    if not only_these_options or "no-trackingstats" in only_these_options:
        parser.add_argument(
            '--plot-tracking-stats', action='store_true', default=False,
            help='plot tracking length distribution for all flies in h5 file (takes some time)')
    if not only_these_options or "portrait" in only_these_options:
        parser.add_argument(
            '--portrait', action='store_true', default=False,
            help='arrange subplots in portrait orientation (one col, many rows)')
    if not only_these_options or "zfilt" in only_these_options:
        parser.add_argument(
            '--zfilt', type=str, choices=filt_choices,
            required=True,
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
            '--basedir', type=str,
            help='base directory in which data files can be found by UUID', default=None)
    if not only_these_options or "outdir" in only_these_options:
        parser.add_argument(
            '--outdir', type=str, default=None,
            help='directory to save plots')
    if not only_these_options or "rfilt" in only_these_options:
        parser.add_argument(
            '--rfilt', type=str, choices=filt_choices,
            required=True,
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

    return parser

def check_args(parser, args):
    if args.uuid:
        if None not in (args.csv_file, args.h5_file):
            parser.error("if uuid is given, --csv-file and --h5-file are not required")
        if len(args.uuid) > 1 and args.outdir is None:
            parser.error("if multiple uuids are given, --outdir is required")
    else:
        if None in (args.csv_file, args.h5_file):
            parser.error("both --csv-file and --h5-file are required")


