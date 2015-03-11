import shutil
import os.path
import argparse
import shutil

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')

import autodata.files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-t', '--type', required=True,
        help='delete the plots from this experiment type')
    parser.add_argument(
        '--uuid', type=str, nargs='*', required=True,
        help='experiment uuids')
    parser.add_argument('-d','--dry-run', default=False, action='store_true',
        help='dont delete directories, print what would be deleted')

    args = parser.parse_args()

    if not args.type.endswith('.py'):
        parser.error("--type must end with '.py'")

    for u in args.uuid:
        fm = autodata.files.FileModel()
        fm.select_uuid(u)

        adir = os.path.join(fm.get_plot_dir(), args.type)

        if os.path.isdir(adir):
            if not args.dry_run:
                shutil.rmtree(adir)
                print "Removed %s" % adir
            print "Will remove %s" % adir


