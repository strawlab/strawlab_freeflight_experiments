from glob import glob
import hashlib
import os
import os.path as op
from strawlab.constants import ensure_dir

from whatami import id2what
from whatami.whatutils import oldid2what


# --- Make old cache available with new whatami4 based ids


def oldwhatami2new():
    """
    Maps the central-combine-cache files (old whatami ids, all in one dir) to the next version
    (whatami 4+ id strings, two-level directory hierarchy).
    """

    def old2new(cached='00027e00204977eda17ac83d8fa47f7b001b81b56c711c9a77cd7545',
                combine_cache_dir='/mnt/strawscience/data/auto_pipeline/cached/combine/'):

        # Check pickle
        pkl = op.join(combine_cache_dir, '%s.pkl' % cached)
        if not op.isfile(pkl):
            raise IOError('Missing pkl file %s' % pkl)

        # Read old configuration
        txt = op.join(combine_cache_dir, '%s.txt' % cached)
        if not op.isfile(txt):
            if 'arena=' not in pkl:
                raise IOError('Missing txt or wrong config string for %s' % pkl)
            txt = op.basename(op.splitext(txt)[0])
        else:
            with open(txt) as reader:
                txt = reader.read()

        # Fix index=blah
        txt = txt.replace('index=time+10L', "index='time+10L'")

        # Make a valid whatami id
        whatdid = 'CombineH5WithCSV#%s' % txt if not txt.startswith('Combine') else txt

        # Parse the id
        what = oldid2what(whatdid) if whatdid.startswith('CombineH5WithCSV#') else id2what(whatdid)

        # Back to new id
        whatid = what.id()

        # Hash it
        whatid_hash = hashlib.sha224(whatid).hexdigest()

        # Shard, symlink, create new txt
        dest = ensure_dir(op.join(combine_cache_dir, whatid_hash[:2]))
        dest_pkl = op.join(dest, whatid_hash + '.pkl')
        if not op.isfile(dest_pkl):
            os.symlink(op.join('..', op.basename(pkl)), dest_pkl)
        dest_txt = op.join(dest, whatid_hash + '.txt')
        with open(dest_txt, 'w') as writer:
            writer.write(whatid)

    def olds2news(combine_cache_dir='/mnt/strawscience/data/auto_pipeline/cached/combine/'):
        cacheds = sorted([op.basename(op.splitext(pkl)[0])
                          for pkl in glob(op.join(combine_cache_dir, '*.pkl'))])
        for cached in cacheds:
            try:
                old2new(cached=cached, combine_cache_dir=combine_cache_dir)
            except Exception as ex:
                print('Trouble with %s: %s' % (cached, str(ex)))

    olds2news()

if __name__ == '__main__':
    oldwhatami2new()

#
# LOG
#
# ... attention please, faked roslib ...
# Trouble with 00027e00204977eda17ac83d8fa47f7b001b81b56c711c9a77cd7545: [Errno 17] File exists
# Trouble with 000e996faed26da7b6a0f0fec7ffe9f71c4344570d52eb4c47e00cc6: Missing txt or wrong config string for /mnt/strawscience/data/auto_pipeline/cached/combine/000e996faed26da7b6a0f0fec7ffe9f71c4344570d52eb4c47e00cc6.pkl
# Trouble with 095a908e94133920970a8b001c38bbdcdfbf7d36272025bcb5aca904: [Errno 17] File exists
# Trouble with 099c94181d4e56677b29e891c56e1c54d75d1d385038a281bdc053b3: Missing txt or wrong config string for /mnt/strawscience/data/auto_pipeline/cached/combine/099c94181d4e56677b29e891c56e1c54d75d1d385038a281bdc053b3.pkl
# Trouble with 0a7eaf31163e5d07371ca6e50c0f65367d1997629239a9f0e1a6e489: Missing txt or wrong config string for /mnt/strawscience/data/auto_pipeline/cached/combine/0a7eaf31163e5d07371ca6e50c0f65367d1997629239a9f0e1a6e489.pkl
# Trouble with 15ccdc2cc605095eef861d40c07a9c7d4fe7f4b7263acb093f180950: Missing txt or wrong config string for /mnt/strawscience/data/auto_pipeline/cached/combine/15ccdc2cc605095eef861d40c07a9c7d4fe7f4b7263acb093f180950.pkl
# Trouble with 18f05cf7d08f229c859e0f4b76e925b1c960162d4f8901400b0b6ab7: Missing txt or wrong config string for /mnt/strawscience/data/auto_pipeline/cached/combine/18f05cf7d08f229c859e0f4b76e925b1c960162d4f8901400b0b6ab7.pkl
# Trouble with 1a62b09290504a950ea677fe3977224c44f4cef6c24a54041402b1b3: Missing txt or wrong config string for /mnt/strawscience/data/auto_pipeline/cached/combine/1a62b09290504a950ea677fe3977224c44f4cef6c24a54041402b1b3.pkl
# Trouble with 369c487de5d2bbb2709904c1486ae7f7a31bfff7e3e0c6b2fda9f093: Missing txt or wrong config string for /mnt/strawscience/data/auto_pipeline/cached/combine/369c487de5d2bbb2709904c1486ae7f7a31bfff7e3e0c6b2fda9f093.pkl
# Trouble with 3f01204ce949bbb948d9aef29b0d18174687db72605024446a34c666: Missing txt or wrong config string for /mnt/strawscience/data/auto_pipeline/cached/combine/3f01204ce949bbb948d9aef29b0d18174687db72605024446a34c666.pkl
# Trouble with 4fa2e629546a1d330f7cc44cba382d1106c842d39235c9a97d100682: Missing txt or wrong config string for /mnt/strawscience/data/auto_pipeline/cached/combine/4fa2e629546a1d330f7cc44cba382d1106c842d39235c9a97d100682.pkl
# Trouble with 58f31a5c725579e543fbe370b77ec4040f7704b2ebf15b954cc7ef8f: Missing txt or wrong config string for /mnt/strawscience/data/auto_pipeline/cached/combine/58f31a5c725579e543fbe370b77ec4040f7704b2ebf15b954cc7ef8f.pkl
# Trouble with 7f4f6569d8b887359d378b4d659edd4a764c8a71014f548b00da8ade: [Errno 17] File exists
# Trouble with 866cf067eb46c3b02e06260072a3b6eaf4ec0de19145502302db4153: Missing txt or wrong config string for /mnt/strawscience/data/auto_pipeline/cached/combine/866cf067eb46c3b02e06260072a3b6eaf4ec0de19145502302db4153.pkl
# Trouble with 8a7dd460af1a6b487ece8e89e1c7b013b0ef7bb116564eeb0ada507b: Missing txt or wrong config string for /mnt/strawscience/data/auto_pipeline/cached/combine/8a7dd460af1a6b487ece8e89e1c7b013b0ef7bb116564eeb0ada507b.pkl
# Trouble with 959e05b312a278a55576a331e0c26403c85c2eec04d4044783ac9dd0: Missing txt or wrong config string for /mnt/strawscience/data/auto_pipeline/cached/combine/959e05b312a278a55576a331e0c26403c85c2eec04d4044783ac9dd0.pkl
# Trouble with 9dec9e477f678968321bc95d3796fda50d4a493e12eced5f1f0be484: Missing txt or wrong config string for /mnt/strawscience/data/auto_pipeline/cached/combine/9dec9e477f678968321bc95d3796fda50d4a493e12eced5f1f0be484.pkl
# Trouble with a1d60848806236a3ca24e7fb590aafe22974b1893d1c1b5908fd6b65: [Errno 17] File exists
# Trouble with a9cfa5f1621e4f8ccde7e5fe0b6223ba78113066d9c793320eccb1e4: Missing txt or wrong config string for /mnt/strawscience/data/auto_pipeline/cached/combine/a9cfa5f1621e4f8ccde7e5fe0b6223ba78113066d9c793320eccb1e4.pkl
# Trouble with afeb44f9de16a7cd5a19bfe6dd4b3ebf1d2e8f3e4c08cb83ef6e3e5d: Missing txt or wrong config string for /mnt/strawscience/data/auto_pipeline/cached/combine/afeb44f9de16a7cd5a19bfe6dd4b3ebf1d2e8f3e4c08cb83ef6e3e5d.pkl
# Trouble with b902ed3feb7d8c2891b9ba3d4b7ea06c2a681b2ca90ef427bedbcc32: Missing txt or wrong config string for /mnt/strawscience/data/auto_pipeline/cached/combine/b902ed3feb7d8c2891b9ba3d4b7ea06c2a681b2ca90ef427bedbcc32.pkl
# Trouble with bc70995d618bfaecaa51880235765b56293be7baa9c4da621d8bbdeb: Missing txt or wrong config string for /mnt/strawscience/data/auto_pipeline/cached/combine/bc70995d618bfaecaa51880235765b56293be7baa9c4da621d8bbdeb.pkl
# Trouble with c6b877aaf42f46ee9a1e932f776be8dba6f2d13da5976d9c84211ec2: Missing txt or wrong config string for /mnt/strawscience/data/auto_pipeline/cached/combine/c6b877aaf42f46ee9a1e932f776be8dba6f2d13da5976d9c84211ec2.pkl
# Trouble with c71344681e28a88cb303f4bb7c93ca9004e478f334d9b582c6450b95: Missing txt or wrong config string for /mnt/strawscience/data/auto_pipeline/cached/combine/c71344681e28a88cb303f4bb7c93ca9004e478f334d9b582c6450b95.pkl
# Trouble with c7a38e1f18f8b0e672f52bc6e73d39e99aeeac5918687d91504117bc: Missing txt or wrong config string for /mnt/strawscience/data/auto_pipeline/cached/combine/c7a38e1f18f8b0e672f52bc6e73d39e99aeeac5918687d91504117bc.pkl
# Trouble with c83720a686faef719822a2b722e671bb8bd4e0a45a82d1b7dc795922: Missing txt or wrong config string for /mnt/strawscience/data/auto_pipeline/cached/combine/c83720a686faef719822a2b722e671bb8bd4e0a45a82d1b7dc795922.pkl
# Trouble with ce0ce405203c098324956c362429745290fcce4ceddc5c5be6db4f91: Missing txt or wrong config string for /mnt/strawscience/data/auto_pipeline/cached/combine/ce0ce405203c098324956c362429745290fcce4ceddc5c5be6db4f91.pkl
# Trouble with ce6a1669b400063250ef02b1d980a4c7abd7d63f9a2330b4452c8dcb: Missing txt or wrong config string for /mnt/strawscience/data/auto_pipeline/cached/combine/ce6a1669b400063250ef02b1d980a4c7abd7d63f9a2330b4452c8dcb.pkl
# Trouble with ceb9e32ee0561a8c21adbff4b89e23f974351f99b74ccb8423429e44: Missing txt or wrong config string for /mnt/strawscience/data/auto_pipeline/cached/combine/ceb9e32ee0561a8c21adbff4b89e23f974351f99b74ccb8423429e44.pkl
# Trouble with db9884228a2766d9e77accbf7b24360bf921ed61d7dbf0fd8f08fc8e: Missing txt or wrong config string for /mnt/strawscience/data/auto_pipeline/cached/combine/db9884228a2766d9e77accbf7b24360bf921ed61d7dbf0fd8f08fc8e.pkl
# Trouble with dd906277343d878b44314f5c89a5a309ede3a44c6cd4f4a229c29f81: [Errno 17] File exists
# Trouble with ec13354b773b9bf140eebae5a484adaa46dfb29e66dd236b07b303d6: Missing txt or wrong config string for /mnt/strawscience/data/auto_pipeline/cached/combine/ec13354b773b9bf140eebae5a484adaa46dfb29e66dd236b07b303d6.pkl
# Trouble with f69224d3c5ac3e17be001769fcbaf91ff40c92a8123e9e6ce3a243be: Missing txt or wrong config string for /mnt/strawscience/data/auto_pipeline/cached/combine/f69224d3c5ac3e17be001769fcbaf91ff40c92a8123e9e6ce3a243be.pkl
# Trouble with faf4f8a31e8711fbdf4a6ca70fbc9aca4e5262f43827b4d6b3841808: Missing txt or wrong config string for /mnt/strawscience/data/auto_pipeline/cached/combine/faf4f8a31e8711fbdf4a6ca70fbc9aca4e5262f43827b4d6b3841808.pkl
# Trouble with fe298c10e108f189583046dc6da35899a985d48d4c282404958e9f68: [Errno 17] File exists
#
# --- / Make old cache available with new whatami4 based ids
#
