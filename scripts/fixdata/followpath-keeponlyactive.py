import os.path
import sys
import pickle
import re
import pprint

sys.path.append('../nodes')
import followpath

import roslib; roslib.load_manifest('strawlab_freeflight_experiments')

try:
    csv_fname = sys.argv[1]
except IndexError:
    csv_fname = ""
finally:
    if not os.path.isfile(csv_fname):
        raise ValueError("no such file")

infile = followpath.Logger(fname=csv_fname, mode="r")
outfile = followpath.Logger(fname=csv_fname+".new", mode="w")

for row in infile.record_iterator():
    try:
        if int(row.active) == 1:
            outfile.write_record(row)
    except ValueError:
        print row

outfile.close()

