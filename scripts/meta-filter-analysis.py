#!/usr/bin/env python2
import sys
import roslib
roslib.load_manifest('strawlab_freeflight_experiments')

import analysislib.args
import analysislib.util as autil


if __name__=='__main__':
    parser = analysislib.args.get_parser()

    args = parser.parse_args()

    analysislib.args.check_args(parser, args)

    combine = autil.get_combiner_for_args(args)
    combine.add_from_args(args)

    results,dt = combine.get_results()

    name = combine.get_plot_filename("FILTER_ANALYSIS.md")
    with open(name, 'w') as f:
        l = "effect of filters on flight data"
        f.write("%s\n"%l)
        f.write("%s\n\n"%('-'*len(l)))
        for a in analysislib.args.REQUIRED_ARENA_DEFAULTS:
            f.write(" * %s = %s\n" % (a,getattr(args,a)))
        f.write("\n\n")
        f.write("| condition | total trials | kept trials | skipped trials | kept data (s) |\n")
        f.write("| --- | --- | --- | --- | --- |\n")

        for cond,r in results.iteritems():
            scond = combine.get_condition_name(cond)
            scond = scond.replace('|','&#124;') #escape | for markdown

            dur = sum(len(df) for df in r['df'])*dt            

            f.write("| %s | %d | %d (%.1f%%) | %d | %.1f |\n" % (scond,
                                                     combine.get_num_trials(cond),
                                                     combine.get_num_analysed(cond),
                                                     100.0 * combine.get_num_analysed(cond) / combine.get_num_trials(cond),
                                                     combine.get_num_skipped(cond),
                                                     dur))

        f.write("\n")

    if args.show:
        print open(name,'r').read()

    sys.exit(0)
