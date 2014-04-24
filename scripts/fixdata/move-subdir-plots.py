#!/usr/bin/env python
import logging
import argparse
import datetime
import os.path
import re
import time
import shlex
import json

import sh

import roslib; roslib.load_manifest('flycave')
import autodata.model
import autodata.files
import strawlab.constants

EXCEPT = set()

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '-t', '--type', required=True,
    help='analyse only this type', choices=["all","perturbation","rotation","confinement","conflict"])
parser.add_argument(
    '-s', '--start', required=True, metavar="2013/03/21",
    help='date from which to rerun analysis scripts')
parser.add_argument(
    '-d', '--dry-run', action='store_true', default=False)
parser.add_argument(
    '-n', '--db-name', default="freeflight",
    help='dbname')
parser.add_argument(
    '-p', '--prefix', default="/flycave/",
    help='prefix')
args = parser.parse_args()

try:
    datetime.datetime.strptime(args.start, "%Y/%m/%d")
except ValueError:
    parser.error("could not parse start date %s" % args.start)

model = autodata.model.SQLModel(
                    **strawlab.constants.get_autodata_connection(
                                            dbname=args.db_name,
                                            prefix=args.prefix))
model.select(start=args.start)

todo = []
for res in model.query(model.table('experiment'), distinct_on="start_secs"):
    if args.type == "all" or res.title.startswith(args.type):
        todo.append(res.uuid)

for uuid in todo:
    if not os.path.isdir("LOG"):
        os.makedirs("LOG")

    if uuid in EXCEPT:
        continue

    fm = autodata.files.FileModel(uuid=uuid)

    how = 'no readme nor json'
    try:
        readme = open(fm.get_output_file("README").fullpath).read()
        how = 'readme'
    except autodata.files.NoFile, e:
        readme = ""

    try:
        jsondata = json.loads(open(fm.get_output_file("data.json").fullpath).read())
        how = 'json'
    except autodata.files.NoFile, e:
        jsondata = None

    #if the experiment contains neither json nor readme, use the default args
    analysis_type = None
    if jsondata:
        match = re.search("(.*-analysis.*.py)", jsondata['argv'])
    elif readme:
        match = re.search("(.*-analysis.*.py)", readme)
    else:
        match = None

    if match:
        analysis_type = str(match.groups()[0])

    if analysis_type:
        old_dir = fm.get_plot_dir()
        new_dir = os.path.join(old_dir, os.path.basename(analysis_type))

        print "---%s\n%s\n->\n%s" % (uuid,old_dir,new_dir)

        if not args.dry_run:
            if not os.path.isdir(new_dir):
                os.makedirs(new_dir)

            for ext in ("png","pkl","json","md"):

                try:
                    sh.mv(sh.glob("%s/*.%s" % (old_dir,ext)),
                          new_dir)

                    print "\tOK %s" % ext

                except sh.ErrorReturnCode:
                    print "\tNO %s" % ext

            for other in ("README",):
                of = os.path.join(old_dir,other)
                if os.path.exists(of):
                    sh.mv(of, new_dir)
                    print "\tOK %s" % other

    else:
        print uuid, "NO ANALYSIS", fm.get_plot_dir(), how



