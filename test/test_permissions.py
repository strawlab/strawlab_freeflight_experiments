#!/usr/bin/env python
import os.path
import unittest
import stat
import random

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')
import analysislib
import strawlab.constants

def isgroup(filepath, mode):
    st = os.stat(filepath)
    return bool(st.st_mode & mode)

def isgroupreadable(filepath):
    return isgroup(filepath,stat.S_IRGRP)

def isgroupwritable(filepath):
    return isgroup(filepath,stat.S_IWGRP)

class TestPermissions(unittest.TestCase):

    def setUp(self):
        strawlab.constants.set_permissions()

    def testPrimaryGroup(self):
        self.assertEqual(os.getgid(), 2046)

    def testDirPermissions(self):
        tdir = os.path.join(strawlab.constants.AUTO_DATA_MNT,'.test')
        self.assertTrue(isgroupreadable(tdir))
        self.assertTrue(isgroupwritable(tdir))

        try:
            fn = os.path.join(tdir,str(random.random()))
            os.mkdir(fn)
            self.assertTrue(isgroupreadable(fn))
            self.assertTrue(isgroupwritable(fn))
            fnf = os.path.join(fn,"TEST")
            with open(fnf,'w') as f:
                f.write("test")
            self.assertTrue(isgroupreadable(fnf))
            self.assertTrue(isgroupwritable(fnf))
        except OSError:
            self.fail('error creating test directory: %s' % fn)

        try:
            os.remove(fnf)
            os.rmdir(fn)
        except OSError:
            self.fail('error removing test directory: %s' % fn)

if __name__=='__main__':
    unittest.main()

