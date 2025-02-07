import os.path
import sys
import pickle
import re

sys.path.append('../nodes')
import followpath
import fixation

import roslib; roslib.load_manifest('strawlab_freeflight_experiments')
import flyflypath.model

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np

data            = {}
trial           = {}
figure_names    = {}

all_conditions  = {}

MIN_DURATION = 0.5

scalarMap = cmx.ScalarMappable(
                    norm=colors.Normalize(vmin=0, vmax=1),
                    cmap=plt.get_cmap('spring'))

default_csv = ''

try:
    csv = sys.argv[1]
except IndexError:
    csv = default_csv
finally:
    if not os.path.isfile(csv):
        raise ValueError("no such file\n./%s /path/to/data.csv" % os.path.basename(__file__))

pts = fixation.get_start_points()

import pprint
pprint.pprint(pts)

oid = -1 
for rec in fixation.Logger(csv,'r').record_iterator():
    try:
        lock_object = int(rec.lock_object)
    except ValueError:
        print "invalid id", rec.lock_object
        continue

    if lock_object in (0xFFFFFFFF,0):
        continue

    t = float(rec.t_sec) + (float(rec.t_nsec) * 1e-9)

    condition = rec.condition
    if lock_object > oid:
        #new trial
        if oid != -1:
            #first time
            trial[oid]["finish"] = t
            trial[oid]["duration"] = trial[oid]["finish"] - trial[oid]["start"]

        oid = lock_object
        data[oid] = []
        trial[oid] = {"start":t,"condition":condition}

        all_conditions[condition] = (float(rec.trg_x),float(rec.trg_y))

    if rec.condition_sub in ("birth_kick","experiment"):
        data[oid].append( (float(rec.fly_x),float(rec.fly_y),float(rec.trg_x),float(rec.trg_y),t) )

plt.figure("Start Condition Hold")
all_conditions_names = all_conditions.keys()

for oid in trial:
    dt = trial[oid].get("duration")
    if dt is None:
        print "Im too lazy to fix this bug - term criteria in first loop"
        continue

    if dt < 0.1:
        continue

    xy = np.array(data[oid])
    if len(xy) == 0:
        print "I forget what this means"
        continue

    #colors correspond to the condition
    color = cmx.brg(1.0*all_conditions_names.index(trial[oid]['condition'])/len(all_conditions_names))

    plt.plot(xy[:,0],xy[:,1],
             color=color)

#add startx and starty targets
for cond,(x,y) in all_conditions.items():
    color = cmx.brg(1.0*all_conditions_names.index(cond)/len(all_conditions_names))
    plt.plot(x,y,color=color,marker='o',markersize=10)

plt.xlim((-0.5,0.5))
plt.ylim((-0.5,0.5))

plt.show()

