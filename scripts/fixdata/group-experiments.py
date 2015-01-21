#!/usr/bin/env python
import logging
import argparse
import datetime
import os.path
import re
import time
import shlex
import json
import collections
import itertools

import sh

import roslib; roslib.load_manifest('flycave')
import autodata.model
import autodata.files
import strawlab.constants

Exp = collections.namedtuple("Exp","uuid genotype conditions")

def run_analysis(db_name, db_prefix, arena, analysis_script, what, args):

    def _print(m):
        if not args.quiet:
            print m


    desc = "%s_%s" % (db_name, db_prefix.replace('/',''))
    _print("\nEXPERIMENTS %s" % desc)

    model = autodata.model.SQLModel(
                **strawlab.constants.get_autodata_connection(
                    dbname=db_name,prefix=db_prefix))
    model.select(start=args.start)

    try:
        todo = []
        for res in model.query(model.table('experiment'), distinct_on="start_secs"):
            if res.title.startswith(args.type):
                todo.append( (res.uuid, res.genotype.lower()) )
    except KeyError:
        model.close()
        _print(desc)
        return [],[]

    results = []
    unique = []
    for uuid,genotype in todo:

        _print("%s %s %s" % (desc,uuid,genotype))
        conds = []
        
        if what == 'condition':
            try:
                filename = os.path.join(
                                autodata.files.FileModel(uuid=uuid).get_plot_dir(),
                                analysis_script, "data.json")
                jsondata = json.loads(open(filename).read())
            except (autodata.files.NoFile, IOError):
                jsondata = None

            if jsondata is not None:
                for c in jsondata['conditions']:
                    _print("\t%s" % c)
                    unique.append( c )
                    conds.append( c )
        elif what == 'genotype':
            unique.append(genotype)

        results.append( Exp(uuid, genotype, conds) )

    model.close()

    return unique, results

def guess_details_from_assay(assay):
    #returns db_name, db_prefix, arena
    assay = assay.strip().replace('/','')
    if assay.startswith('flycube'):
        return "flycube", "/%s/" % assay, "flycube"
    elif assay == "flycave":
        return "freeflight", "/flycave/", "flycave"
    elif assay in ("fishvr","fishtrax"):
        return "fishvr", "/fishtrax/", "fishtrax"
    else:
        raise Exception("Unknown assay %s" % assay)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-t', '--type', required=True,
        help='consider only this experiment type (the name of the experiment in the database)')
    parser.add_argument(
        '-a', '--analysis-type',
        help='run this analysis script (e.g. rotation, translation, rotation-flycube)')
    parser.add_argument(
        '-s', '--start', required=True, metavar="2013/03/21",
        help='date from which to rerun analysis scripts')
    parser.add_argument(
        '-A', '--arena', default='flycave',
        help='flycave, flycube, etc')
    parser.add_argument(
        '-N', '--db-name', default="freeflight",
        help='database name for listing experiments')
    parser.add_argument(
        '-P', '--db-prefix', default="/flycave/",
        help='database prefix (assay name) for listing experiments')
    parser.add_argument(
        '-S','--assay',
        help='the name of the assay (flycave, flycube4, fishtrax, etc). Pass this '
             'instead of --db-name --db-prefix and --assay to automatically guess these '
             'values. Can be a comma separated list of assays to process more data')
    parser.add_argument(
        '-q', '--quiet', action='store_true',
        help='only print uuids according to the grouped value')
    parser.add_argument(
        '-g', '--groupby', choices=['condition','genotype','none'],
        help='print and optionally group by this information in each experiment')
    parser.add_argument(
        '-G', '--groupby-value',
        help='group experiments together by this value (either a genotype or condition string)')


    args = parser.parse_args()

    if args.analysis_type:
        analysis_script = "%s.py" % args.analysis_type
    else:
        analysis_script = "%s-analysis.py" % args.type

    try:
        datetime.datetime.strptime(args.start, "%Y/%m/%d")
    except ValueError:
        parser.error("could not parse start date %s" % args.start)

    results = []
    unique = []
    if args.assay:
        for assay in args.assay.split(','):
            db_name, db_prefix, arena = guess_details_from_assay(assay)
            u,r = run_analysis(db_name, db_prefix, arena, analysis_script, args.groupby, args)
            unique += u
            results += r
    else:
        unique,results = run_analysis(args.db_name, args.db_prefix, args.arena, analysis_script, '', args)

    if not args.quiet:
        print "\nSUMMARY"
        for k,v in collections.Counter(unique).items():
            print "%d\t%s" % (v,k)

    if args.groupby == 'none':
        print " ".join(map(str,(r.uuid for r in results)))

    if args.groupby_value:
        if not args.quiet:
            print "\nAGGREGATION BY %s=%s" % (args.groupby,args.groupby_value)

        v = args.groupby_value.lower()
        if args.groupby == 'genotype':
            uuids = [r.uuid for r in itertools.ifilter(lambda x: x.genotype == v, results)]
            print " ".join(map(str,uuids))
        elif args.groupby == 'condition':
            if args.quiet:
                uuids = [r.uuid for r in itertools.ifilter(lambda x: v in x.conditions, results)]
                print " ".join(map(str,uuids))
            else:
                for r in results:
                    if v in r.conditions:
                        print "%s #%s" % (r.uuid, r.genotype)



