#!/usr/bin/env python
import logging
import argparse
import datetime
import os.path
import re
import time
import shlex
import json
import webbrowser

import sh

import roslib; roslib.load_manifest('flycave')
import autodata.model
import autodata.files
import strawlab.constants

DEFAULT_LENFILT = '--lenfilt 1'
DEFAULT_ARGS    = '--uuid %s --zfilt trim --rfilt trim ' + DEFAULT_LENFILT + ' --arena %s'

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
        if args.force_args or (not jsondata and not readme):
            opts = DEFAULT_ARGS % (uuid, arena)
            where = 'f' if args.force_args else 'd' 
        elif jsondata:
            opts = jsondata['argv'].split('.py ')[1]
            where = 'j'
        elif readme:
            match = re.search(r"""--lenfilt[ ]*?([1-9]+)""", readme)
            if match:
                lenfilt = str(match.groups()[0])
            else:
                lenfilt = "1"
            opts = DEFAULT_ARGS % (uuid, arena)
            opts = opts.replace(DEFAULT_LENFILT, '--lenfilt %s' % lenfilt)
            where = 'r'
        else:
            raise Exception("Error generating arguments")

        print where,

        t = time.time()
        try:
            if args.no_rosrun:
                argslist = [analysis_script]
            else:
                argslist = ["strawlab_freeflight_experiments", analysis_script]
            for opt in opts.split(' '):
                if opt.strip() == '--no-reindex':
                    if args.no_reindex:
                        argslist.append('--no-reindex')
                elif opt.strip() == '--cached':
                    if args.no_cached:
                        argslist.append('--no-cached')
                elif opt.strip() == '--show':
                    pass
                #don't retain backwards compat, its too hard and not worthwhile
                elif opt.strip() == '--reindex':
                    pass
                elif opt.strip() == '--cached':
                    pass
                else:
                    argslist.append(opt)

            if args.extra_args:
                for opt in args.extra_args.split(' '):
                    argslist.append(opt)

            if not args.dry_run:
                if args.no_rosrun:
                    sh.python(
                    *argslist,
                    _out=os.path.join("LOG","%s.stdout" % uuid),
                    _err=os.path.join("LOG","%s.stderr" % uuid))
                else:
                    sh.rosrun(
                        *argslist,
                        _out=os.path.join("LOG","%s.stdout" % uuid),
                        _err=os.path.join("LOG","%s.stderr" % uuid))

            dt = time.time() - t
            print "succeeded (%.1fs)" % dt, " ".join(argslist)

            if args.open:
                try:
                    webbrowser.open_new_tab(strawlab.constants.get_experiment_result_url(uuid))
                except:
                    pass

        except Exception, e:
            print "failed", opts, e

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
        '-f', '--force-args', action='store_true',
        help='rerun analysis with the default arguments irrespective of those run previously')
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
        '-n', '--no-reindex', action='store_true',
        help='dont reindex h5 file')
    parser.add_argument(
        '-c', '--no-cached', action='store_true',
        help='dont used cached data')
    parser.add_argument(
        '-S','--assay',
        help='the name of the assay (flycave, flycube4, fishtrax, etc). Pass this '
             'instead of --db-name --db-prefix and --assay to automatically guess these '
             'values. Can be a comma separated list of assays to process more data')
    parser.add_argument(
        '-r', '--no-rosrun', action='store_true', default=False)
    parser.add_argument(
        '-o', '--open', action='store_true',
        help='open results in webbrowser when analysis completes successfully')

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

