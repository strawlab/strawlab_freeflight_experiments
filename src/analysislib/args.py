import os.path
import argparse

import roslib
roslib.load_manifest('flycave')
import autodata.files

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--csv-file', type=str)
    parser.add_argument(
        '--h5-file', type=str)
    parser.add_argument(
        '--hide-obj-ids', action='store_false', dest='show_obj_ids', default=True)
    parser.add_argument(
        '--show', action='store_true', default=False)
    parser.add_argument(
        '--zfilt', action='store_true', default=False)
    parser.add_argument(
        '--zfilt-min', type=float, default=0.10)
    parser.add_argument(
        '--zfilt-max', type=float, default=0.90)
    parser.add_argument(
        '--uuid', type=str,
        help='get the appropriate csv and h5 file for this uuid')

    return parser

def parse_csv_and_h5_file(parser, args, csv_suffix):
    if args.uuid:
        if None not in (args.csv_file, args.h5_file):
            parser.error("if uuid is given, --csv-file and --h5-file are not required")
        fm = autodata.files.FileModel()
        fm.select_uuid(args.uuid)
        csv_file = fm.get_csv(csv_suffix).fullpath
        h5_file = fm.get_simple_h5().fullpath
    else:
        if None in (args.csv_file, args.h5_file):
            parser.error("both --csv-file and --h5-file are required")
        csv_file = args.csv_file
        h5_file = args.h5_file

    print csv_file
    print h5_file

    assert os.path.exists(csv_file)
    assert os.path.exists(h5_file)

    return csv_file, h5_file
