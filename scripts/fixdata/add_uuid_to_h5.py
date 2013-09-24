import argparse
import sys
import os.path
import uuid
import flydra.data_descriptions
import tables as PT
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

def add_uuid_to_h5(h5,u):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--uuid', type=str, required=False)
    args = parser.parse_args()
    if not os.path.exists(args.file):
        parser.error("file not found")

    if args.uuid:
        add_uuid_to_h5(args.file, args.uuid)
    else:
        read_uuid_from_h5(args.file)
