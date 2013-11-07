import argparse

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')
import analysislib.args
import analysislib.util as autil

def my_function(dt):
    print "dt is", dt

if __name__ == "__main__":
    parser = analysislib.args.get_parser(
                    zfilt='trim',
                    rfilt='trim',
    )
    #add your own optios here
    parser.add_argument(
        "--do-my-function", action="store_true",
        help="do my awesome analysis"
    )
   
    args = parser.parse_args()

    #this checks that the user has specified enough UUIDs or csv and h5 file
    analysislib.args.check_args(parser, args, max_uuids=1)

    combine = autil.get_combiner(autil.get_csv_for_args(args))
    #dont print to the console as we load the file
    combine.disable_debug()

    combine.add_from_args(args)

    #get the data as requested on the command line
    results,dt = combine.get_results()

    #do your own analysis
    if args.do_my_function:
        my_function(dt)

