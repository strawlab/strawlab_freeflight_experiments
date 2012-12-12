import argparse
import sys
import os.path
import uuid

def add_uuid_to_csv(csv,u):
    assert os.path.exists(csv)
    with open(csv,'r') as fi:
        with open(csv+'.new','w') as fo:
            for i,line in enumerate(fi):
                if i == 0:
                    fo.write(line[0:-1])
                    fo.write(',exp_uuid\n')
                else:
                    fo.write(line[0:-1])
                    fo.write(',')
                    fo.write(u)
                    fo.write('\n')
    return u

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--uuid', type=str, default=uuid.uuid1().get_hex(), required=False)
    args = parser.parse_args()
    print "ADDED UUID:\n\t%s" % add_uuid_to_csv(args.file, args.uuid)
