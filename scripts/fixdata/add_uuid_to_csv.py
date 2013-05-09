import argparse
import sys
import os.path
import uuid

def add_uuid_to_csv(csv,u):
    assert os.path.exists(csv)
    has_col = False
    header = ''
    with open(csv,'r') as fi:
        with open(csv+'.new','w') as fo:
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

    return u

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--uuid', type=str, default=uuid.uuid1().get_hex(), required=False)
    args = parser.parse_args()
    print "ADDED UUID:\n\t%s" % add_uuid_to_csv(args.file, args.uuid)
