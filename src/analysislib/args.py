import os.path
import argparse

import roslib
roslib.load_manifest('flycave')
import autodata.files

from .filters import FILTER_REMOVE, FILTER_TRIM, FILTER_NOOP

def get_parser():
    filt_choices = (FILTER_REMOVE, FILTER_TRIM, FILTER_NOOP)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--csv-file', type=str,
        help='path to *.csv file (if not using --uuid)')
    parser.add_argument(
        '--h5-file', type=str,
        help='path to simple_flydra.h5 file (if not using --uuid)')
    parser.add_argument(
        '--show-obj-ids', action='store_true', default=False,
        help='show obj_ids on plots where appropriate')
    parser.add_argument(
        '--reindex', action='store_true', default=False,
        help='reindex simple_flydra h5 file')
    parser.add_argument(
        '--show', action='store_true', default=False,
        help='show plots')
    parser.add_argument(
        '--no-trackingstats', action='store_true', default=False,
        help='plot tracking length distribution for all flies in h5 file (takes some time)')
    parser.add_argument(
        '--portrait', action='store_true', default=False,
        help='arrange subplots in portrait orientation (one col, many rows)')
    parser.add_argument(
        '--zfilt', type=str, choices=filt_choices,
        required=True,
        help='method to filter trajectory data based on z values')
    parser.add_argument(
        '--zfilt-min', type=float, default=0.10,
        help='minimum z, metres (default %(default)s)')
    parser.add_argument(
        '--zfilt-max', type=float, default=0.90,
        help='maximum z, metres (default %(default)s)')
    parser.add_argument(
        '--uuid', type=str, nargs='*', default=None,
        help='get the appropriate csv and h5 file for this UUID (multiple may be specified)')
    parser.add_argument(
        '--basedir', type=str,
        help='base directory in which data files can be found by UUID', default=None)
    parser.add_argument(
        '--outdir', type=str, default=None,
        help='directory to save plots')
    parser.add_argument(
        '--rfilt', type=str, choices=filt_choices,
        required=True,
        help='method to filter trajectory data based on radius from centre values')
    parser.add_argument(
        '--rfilt-max', type=float, default=0.42,
        help='maximum r, metres, (default %(default)s)')
    parser.add_argument(
        '--lenfilt', type=float, default=1.0,
        required=False,
        help='filter trajectories shorter than this many seconds (default %(default)s)')
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


