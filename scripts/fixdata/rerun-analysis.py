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

DEFAULT_LENFILT = '--lenfilt 1'
DEFAULT_ARGS    = '--uuid %s --zfilt trim --rfilt trim ' + DEFAULT_LENFILT + ' --reindex --arena flycave'

EXCEPT = set()

parser = argparse.ArgumentParser()
parser.add_argument(
    '-t', '--type', required=True,
    help='consider only this experiment type', choices=["rotation","confinement","conflict","perturbation","translation"])
parser.add_argument(
    '-a', '--analysis-type',
    help='run this analysis script', choices=["rotation","confinement","conflict","perturbation","translation"])
parser.add_argument(
    '-s', '--start', required=True, metavar="2013/03/21",
    help='date from which to rerun analysis scripts')
parser.add_argument(
    '-d', '--dry-run', action='store_true', default=False)
parser.add_argument(
    '-e', '--extra-args',
    help='extra args to add to the command')
parser.add_argument(
    '-n', '--no-reindex', action='store_false', default=True, dest='reindex',
    help='dont reindex h5 file')
args = parser.parse_args()

if not args.analysis_type:
    args.analysis_type = args.type

try:
    datetime.datetime.strptime(args.start, "%Y/%m/%d")
except ValueError:
    parser.error("could not parse start date %s" % args.start)

model = autodata.model.SQLModel()
model.select(start=args.start)

todo = []
for res in model.query(model.table('experiment'), distinct_on="start_secs"):
    if res.title.startswith(args.type):
        todo.append(res.uuid)

for uuid in todo:
    if not os.path.isdir("LOG"):
        os.makedirs("LOG")

    if uuid in EXCEPT:
        continue

    print uuid,

    try:
        filename = os.path.join(
                        autodata.files.FileModel(uuid=uuid).get_plot_dir(),
                        '%s-analysis.py' % args.analysis_type,"README")
        readme = open(filename).read()
    except (autodata.files.NoFile, IOError):
        readme = ""

    try:
        filename = os.path.join(
                        autodata.files.FileModel(uuid=uuid).get_plot_dir(),
                        '%s-analysis.py' % args.analysis_type,"data.json")
        jsondata = json.loads(open(filename).read())
    except (autodata.files.NoFile, IOError):
        jsondata = None

    #if the experiment contains neither json nor readme, use the default args
    if not jsondata and not readme:
        opts = DEFAULT_ARGS % uuid
        where = 'd'
    elif jsondata:
        opts = jsondata['argv'].split('analysis.py ')[1]
        where = 'j'
    elif readme:
        match = re.search(r"""--lenfilt[ ]*?([1-9]+)""", readme)
        if match:
            lenfilt = str(match.groups()[0])
        else:
            lenfilt = "1"
        opts = DEFAULT_ARGS % uuid
        opts = opts.replace(DEFAULT_LENFILT, '--lenfilt %s' % lenfilt)
        where = 'r'
    else:
        raise Exception("Error generating arguments")

    print where,

    t = time.time()
    try:
        argslist = ["strawlab_freeflight_experiments", "%s-analysis.py" % args.analysis_type]
        for opt in opts.split(' '):
            if opt.strip() == '--reindex':
                if args.reindex:
                    argslist.append(opt)
            elif opt.strip() == '--cached':
                pass
            elif opt.strip() == '--show':
                pass
            else:
                argslist.append(opt)

        if not args.dry_run:
            sh.rosrun(
                *argslist,
                _out=os.path.join("LOG","%s.stdout" % uuid),
                _err=os.path.join("LOG","%s.stderr" % uuid))

        dt = time.time() - t
        print "succeeded (%.1fs)" % dt, " ".join(argslist[1:])
    except Exception, e:
        print "failed", opts

