import os.path
import argparse

import roslib
roslib.load_manifest('flycave')
import autodata.files

from .filters import FILTER_REMOVE, FILTER_TRIM, FILTER_NOOP

def get_parser(*only_these_options):
    filt_choices = (FILTER_REMOVE, FILTER_TRIM, FILTER_NOOP)

    parser = argparse.ArgumentParser()
    if only_these_options and "csv-file" in only_these_options:
        parser.add_argument(
            '--csv-file', type=str,
            help='path to *.csv file (if not using --uuid)')
    if only_these_options and "h5-file" in only_these_options:
        parser.add_argument(
            '--h5-file', type=str,
            help='path to simple_flydra.h5 file (if not using --uuid)')
    if only_these_options and "show-obj-ids" in only_these_options:
        parser.add_argument(
            '--show-obj-ids', action='store_true', default=False,
            help='show obj_ids on plots where appropriate')
    if only_these_options and "reindex" in only_these_options:
        parser.add_argument(
            '--reindex', action='store_true', default=False,
            help='reindex simple_flydra h5 file')
    if only_these_options and "show" in only_these_options:
        parser.add_argument(
            '--show', action='store_true', default=False,
            help='show plots')
    if only_these_options and "no-trackingstats" in only_these_options:
        parser.add_argument(
            '--no-trackingstats', action='store_true', default=False,
            help='plot tracking length distribution for all flies in h5 file (takes some time)')
    if only_these_options and "portrait" in only_these_options:
        parser.add_argument(
            '--portrait', action='store_true', default=False,
            help='arrange subplots in portrait orientation (one col, many rows)')
    if only_these_options and "zfilt" in only_these_options:
        parser.add_argument(
            '--zfilt', type=str, choices=filt_choices,
            required=True,
            help='method to filter trajectory data based on z values')
    if only_these_options and "zfilt-min" in only_these_options:
        parser.add_argument(
            '--zfilt-min', type=float, default=0.10,
            help='minimum z, metres (default %(default)s)')
    if only_these_options and "zfilt-max" in only_these_options:
        parser.add_argument(
            '--zfilt-max', type=float, default=0.90,
            help='maximum z, metres (default %(default)s)')
    if only_these_options and "uuid" in only_these_options:
        parser.add_argument(
            '--uuid', type=str, nargs='*', default=None,
            help='get the appropriate csv and h5 file for this UUID (multiple may be specified)')
    if only_these_options and "basedir" in only_these_options:
        parser.add_argument(
            '--basedir', type=str,
            help='base directory in which data files can be found by UUID', default=None)
    if only_these_options and "outdir" in only_these_options:
        parser.add_argument(
            '--outdir', type=str, default=None,
            help='directory to save plots')
    if only_these_options and "rfilt" in only_these_options:
        parser.add_argument(
            '--rfilt', type=str, choices=filt_choices,
            required=True,
            help='method to filter trajectory data based on radius from centre values')
    if only_these_options and "rfilt-max" in only_these_options:
        parser.add_argument(
            '--rfilt-max', type=float, default=0.42,
            help='maximum r, metres, (default %(default)s)')
    if only_these_options and "lenfilt" in only_these_options:
        parser.add_argument(
            '--lenfilt', type=float, default=1.0,
            required=False,
            help='filter trajectories shorter than this many seconds (default %(default)s)')
    if only_these_options and "idfilt" in only_these_options:
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


