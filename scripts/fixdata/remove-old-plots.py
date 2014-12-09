#!/usr/bin/env python
import logging
import argparse
import datetime
import os.path
import re
import time
import shlex
import json
import shutil

import sh

import roslib; roslib.load_manifest('flycave')
import autodata.model
import autodata.files
import strawlab.constants

DEFAULT_LENFILT = '--lenfilt 1'
DEFAULT_ARGS    = '--uuid %s --zfilt trim --rfilt trim ' + DEFAULT_LENFILT + ' --reindex --arena %s'

EXCEPT = set()

def run_analysis(db_name, db_prefix, arena, analysis_script, args):

    desc = "%s_%s" % (db_name, db_prefix.replace('/',''))

    model = autodata.model.SQLModel(
                **strawlab.constants.get_autodata_connection(
                    dbname=db_name,prefix=db_prefix))
    model.select(start=args.start)

    try:
        todo = []
        for res in model.query(model.table('experiment'), distinct_on="start_secs"):
            if res.title.startswith(args.type):
                todo.append(res.uuid)
    except KeyError:
        model.close()
        print desc
        return

    for uuid in todo:
        if not os.path.isdir("LOG"):
            os.makedirs("LOG")

        if uuid in EXCEPT:
            continue

        print desc,uuid,

        plotdir = os.path.join(autodata.files.FileModel(uuid=uuid).get_plot_dir(),
                               analysis_script)
        if os.path.isdir(plotdir):
            if not args.dry_run:
                shutil.rmtree(plotdir)
            print 'rm -rf', plotdir
        else:
            print 'no plots'

    model.close()

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
        '-d', '--dry-run', action='store_true', default=False)
    parser.add_argument(
        '-e', '--extra-args',
        help='extra args to add to the command')
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

    args = parser.parse_args()

    if args.analysis_type:
        analysis_script = "%s.py" % args.analysis_type
    else:
        analysis_script = "%s-analysis.py" % args.type

    try:
        datetime.datetime.strptime(args.start, "%Y/%m/%d")
    except ValueError:
        parser.error("could not parse start date %s" % args.start)

    if args.assay:
        for assay in args.assay.split(','):
            db_name, db_prefix, arena = guess_details_from_assay(assay)
            run_analysis(db_name, db_prefix, arena, analysis_script, args)
    else:
        run_analysis(args.db_name, args.db_prefix, args.arena, analysis_script, args)

