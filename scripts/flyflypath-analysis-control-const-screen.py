import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np
import csv

data = {}
trial = {}

MIN_RATIO = .1

scalarMap = cmx.ScalarMappable(
                    norm=colors.Normalize(vmin=0, vmax=1),
                    cmap=plt.get_cmap('spring'))

with open('DATA20121003_114423.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    first = True
    oid = -1
    for row in reader:
        if first:
            first = False
            continue
        
        svg_filename,condition,src_x,src_y,src_z,target_x,target_y,target_z,stim_x,stim_y,stim_z,move_ratio,active,lock_object,framenumber,t_sec,t_nsec = row

        lock_object = int(lock_object)
        if lock_object > oid and lock_object != 0xFFFFFFFF:
            oid = lock_object
            data[oid] = []
            trial[oid] = condition

        if oid <= 0:
            continue

        if int(active):
            data[oid].append((float(src_x),float(src_y),float(move_ratio)))

for oid in data:
    xy = np.array(data[oid])
    move_ratio = xy.max(axis=0)[2]
    if move_ratio > MIN_RATIO:
        plt.figure(trial[oid])
        plt.xlim((-0.5,0.5))
        plt.ylim((-0.5,0.5))
        plt.plot(xy[:,0],xy[:,1],
                label='%d'%oid,
                color=scalarMap.to_rgba(move_ratio))
        plt.legend(loc='upper right')


plt.show()
