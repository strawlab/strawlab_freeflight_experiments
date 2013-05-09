import argparse
import sys
import os.path
import uuid
import flydra.data_descriptions
import tables as PT

def add_uuid_to_h5(h5,u):
    assert os.path.exists(h5)
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
    return u

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--uuid', type=str, default=uuid.uuid1().get_hex(), required=False)
    args = parser.parse_args()
    print "ADDED UUID:\n\t%s" % add_uuid_to_h5(args.file, args.uuid)
