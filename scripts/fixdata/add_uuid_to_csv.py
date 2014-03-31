#!/usr/bin/env python

import argparse
import sys
import os.path
import uuid
import numpy
import pandas as pd

def read_uuid_from_csv(fname):
    df = pd.read_csv(fname)
    uuids = []
    try:
        uuids = df['exp_uuid'].dropna().unique().tolist()
    except KeyError:
        print "MISSING EXP_UUID COLUMN"
    except Exception, e:
        print "CORRUPT CSV", e

    print "DETECTED UUID:\n\t%r" % uuids
    return uuids

def add_uuid_to_csv(csv,u):
    assert os.path.exists(csv)
    has_col = False
    header = ''
    newcsv = csv+'.new'
    oldcsv = csv+'.old'
    with open(csv,'r') as fi:
        with open(newcsv,'w') as fo:
            for i,line in enumerate(fi):
                if i == 0:
                    #[0:-1] strips the \n
                    has_col = 'exp_uuid' in line
                    header = line[0:-1]
                    if has_col:
                        fo.write(line)
                    else:
                        fo.write(line[0:-1])
                        fo.write(',exp_uuid\n')
                    continue

                if i == 1:
                    if has_col:
                        idx = header.split(',').index('exp_uuid')
                        parts = line[0:-1].split(',')
                        parts[idx] = u
                        fo.write(','.join(parts))
                        fo.write('\n')
                    else:
                        fo.write(line[0:-1])
                        fo.write(',')
                        fo.write(u)
                        fo.write('\n')
                    continue

                if has_col:
                    fo.write(line)
                else:
                    fo.write(line[0:-1])
                    fo.write(',')
                    fo.write(u)
                    fo.write('\n')

    print "ADDED UUID:\n\t%s" % u

    os.rename(csv,oldcsv)
    os.rename(newcsv,csv)

    return u

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--uuid', type=str, required=False,
             help='if not specified, print any existing UUIDs')
    args = parser.parse_args()

    if not os.path.exists(args.file):
        parser.error("file not found")

    if args.uuid:
        add_uuid_to_csv(args.file, args.uuid)
    else:
        print 'no UUID given - not adding UUID'
        read_uuid_from_csv(args.file)

