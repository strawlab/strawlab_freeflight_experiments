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

DEFAULT_LENFILT = '--lenfilt 1'
DEFAULT_ARGS    = '--uuid %s --zfilt trim --rfilt trim ' + DEFAULT_LENFILT + ' --reindex --arena %s'

DEFAULT_ARGS = r"""--uuid %s --zfilt trim --zfilt-min 0.05 --zfilt-max 0.38 --rfilt trim --arena %s --reindex --customfilt "df[df['velocity']>0.1]" --customfilt-len 1"""

EXCEPT = set()

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '-t', '--type', required=True,
    help='consider only this experiment type', choices=["rotation","confinement","conflict","perturbation","translation"])
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
    '-A', '--arena',
    help='flycave, flycube, etc')
parser.add_argument(
    '-N', '--db-name', default="freeflight",
    help='database name for listing experiments')
parser.add_argument(
    '-P', '--db-prefix', default="/flycave/",
    help='database prefix (assay name) for listing experiments')
parser.add_argument(
    '-n', '--no-reindex', action='store_false', default=True, dest='reindex',
    help='dont reindex h5 file')
args = parser.parse_args()

if not args.analysis_type:
    args.analysis_type = args.type

try:
    analysis_type,suffix = args.analysis_type.split('-')
    suffix = '-' + suffix
except ValueError:
    analysis_type = args.analysis_type
    suffix = ''

analysis_script = "%s-analysis%s.py" %(analysis_type,suffix)

try:
    datetime.datetime.strptime(args.start, "%Y/%m/%d")
except ValueError:
    parser.error("could not parse start date %s" % args.start)

model = autodata.model.SQLModel(
            **strawlab.constants.get_autodata_connection(
                dbname=args.db_name,prefix=args.db_prefix))
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
                        analysis_script, "README")
        readme = open(filename).read()
    except (autodata.files.NoFile, IOError):
        readme = ""

    try:
        filename = os.path.join(
                        autodata.files.FileModel(uuid=uuid).get_plot_dir(),
                        analysis_script, "data.json")
        jsondata = json.loads(open(filename).read())
    except (autodata.files.NoFile, IOError):
        jsondata = None

    #if the experiment contains neither json nor readme, use the default args
    if True:#not jsondata and not readme:
        opts = DEFAULT_ARGS % (uuid, args.arena)
        where = 'd'
    elif jsondata:
        opts = jsondata['argv'].split('.py ')[1]
        where = 'j'
    elif readme:
        match = re.search(r"""--lenfilt[ ]*?([1-9]+)""", readme)
        if match:
            lenfilt = str(match.groups()[0])
        else:
            lenfilt = "1"
        opts = DEFAULT_ARGS % (uuid, args.arena)
        opts = opts.replace(DEFAULT_LENFILT, '--lenfilt %s' % lenfilt)
        where = 'r'
    else:
        raise Exception("Error generating arguments")

    print where,

    t = time.time()
    try:
        argslist = ["strawlab_freeflight_experiments", analysis_script]
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

