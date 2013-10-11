#!/usr/bin/env python

import argparse
import sys
import os.path
import uuid
import flydra.data_descriptions
import h5py
import numpy as np

def read_uuid_from_h5(h5):
    uuids = []
    with h5py.File(h5,'r') as f:
        try:
            uuids.extend( str(r[0]) for r in np.unique(f['experiment_info']) )
        except KeyError:
            #no such table
            pass

    print "DETECTED UUID:\n\t%r" % uuids
    return uuids

def add_uuid_to_old_h5(h5,u):
    import tables as PT

    f = PT.openFile(h5, mode="r+")
    ct = f.createTable
    root = f.root
    if 'experiment_info' not in root:
        exp_info = ct(root,'experiment_info',flydra.data_descriptions.ExperimentInfo,"ExperimentInfo",expectedrows=100)
    exp_info = root.experiment_info
    exp_info.row['uuid'] = u
    exp_info.row.append()
    exp_info.flush()
    f.close()

    print "ADDED UUID:\n\t%s" % u
    return u

def is_simple_flydra_h5_file(fname):
    has_trajectories = False
    with h5py.File(fname,'r') as f:
        try:
            if f['trajectories']:
                has_trajectories = True
        except KeyError:
            pass
    return has_trajectories

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--uuid', type=str, required=False,
            help='if not specified, print any existing UUIDS')
    args = parser.parse_args()
    if not os.path.exists(args.file):
        parser.error("file not found")

    is_simple_h5 = is_simple_flydra_h5_file(args.file)
    if is_simple_h5:
        sign = ''
    else:
        sign = 'NOT '

    print ('I think this is %sa simple h5 file as described at '
           'http://strawlab.org/schemas/flydra/1.1/' % sign)

    if args.uuid:

        if is_simple_h5:
            if 1:
                raise NotImplementedError('need to write pytables-free add_uuid_to_simple_h5()')
            add_uuid_to_simple_h5(args.file, args.uuid)
        else:
            add_uuid_to_old_h5(args.file, args.uuid)
    else:
        read_uuid_from_h5(args.file)
