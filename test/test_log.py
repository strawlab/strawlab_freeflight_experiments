#!/usr/bin/env python
import unittest
import tempfile
import shutil
import os.path
import threading

import pandas as pd

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')
import nodelib.log
import strawlab_freeflight_experiments.conditions as sfe_cond

TEST_STATE = {
"a_float":float,
"b_int":int,
"c_string":str,
}
TEST_STATE_NAMES = sorted(TEST_STATE.keys())

class Logger(nodelib.log.CsvLogger):
    STATE = TEST_STATE_NAMES

class TestLog(unittest.TestCase):

    def setUp(self):
        self._tdir = tempfile.mkdtemp()
        self._fn = os.path.join(self._tdir,"log.csv")

    def test_attempt_read_extra_rows(self):
        N = 10

        def _write():
            l = nodelib.log.CsvLogger(self._fn,'w',state=TEST_STATE_NAMES)
            for i in range(N):
                for s in TEST_STATE_NAMES:
                    setattr(l,s,TEST_STATE[s](i))
                l.update()
            return l.close()

        #extra row definitions are ignored, record_iterator returns row objects
        #with all cols set by default
        l = nodelib.log.CsvLogger(_write(),'r',state=TEST_STATE_NAMES+['extra'])

        i = 0
        for row in l.record_iterator():
            for s in TEST_STATE_NAMES:
                self.assertEqual(TEST_STATE[s](i), TEST_STATE[s](getattr(row,s)))
            i += 1
        self.assertEqual(i,N)

        #state is even optional for readers
        l = nodelib.log.CsvLogger(_write(),'r')

        i = 0
        for row in l.record_iterator():
            for s in TEST_STATE_NAMES:
                self.assertEqual(TEST_STATE[s](i), TEST_STATE[s](getattr(row,s)))
            i += 1
        self.assertEqual(i,N)


    def test_constuct(self):
        fn = os.path.join(self._tdir,"log.csv")
        l = Logger(fn, 'w')

        cols = l.columns
        for s in list(TEST_STATE_NAMES) + list(nodelib.log.CsvLogger.EXTRA_STATE):
            self.assertTrue(s in cols)
        self.assertEqual(l.close(), fn)

        l = nodelib.log.CsvLogger(fn,'w',state=TEST_STATE_NAMES)
        self.assertEqual(l.close(), fn)

        cols = l.columns
        for s in list(TEST_STATE_NAMES) + list(nodelib.log.CsvLogger.EXTRA_STATE):
            self.assertTrue(s in cols)
        self.assertEqual(l.close(), fn)

    def test_comma(self):
        l = nodelib.log.CsvLogger(self._fn,'w', state=('a_int',))

        c = sfe_cond.Condition({'data':'something,with,[weird,{charact ers  ,,;:'})
        c.name = 'test_cond'
        l.condition = c

        l.a_int = 5

        l.update()
        fn2 = l.close()

        #2 lines in file, second is known
        with open(fn2) as f:
            for s in f.readlines():
                pass
        self.assertTrue(s.startswith('5,"something,with,[weird,{charact ers  ,,;:",test_cond'))

        df = pd.read_csv(fn2)
        self.assertListEqual(df.columns.values.tolist(), ['a_int', 'condition', 'condition_name', 'lock_object', 'framenumber', 't_sec', 't_nsec', 'flydra_data_file', 'exp_uuid'])

        self.assertEqual(len(df),1)
        self.assertEqual(df['condition'].unique()[0],c['data'])

    def test_read(self):
        l = nodelib.log.CsvLogger(self._fn,'w',state=TEST_STATE_NAMES)
        for s in TEST_STATE_NAMES:
            setattr(l,s,TEST_STATE[s](50))
        l.update()
        fn2 = l.close()
        self.assertEqual(self._fn,fn2)

        HEAD = 'a_float,b_int,c_string,condition,condition_name,lock_object,framenumber,t_sec,t_nsec,flydra_data_file,exp_uuid\n'

        with open(fn2) as f:
            header,line = f.readlines()
            self.assertEqual(header, HEAD)
            self.assertTrue(line.startswith('50.0,50,50,"",None,None,None,'))

        #try duplicate csv row
        l = nodelib.log.CsvLogger(self._fn+'.2','w',state=TEST_STATE_NAMES+['framenumber'])
        l.update()
        fn2 = l.close()

        with open(fn2) as f:
            header,line = f.readlines()
            self.assertEqual(header, HEAD)

    def test_write_many_read(self):
        N = 10000

        fn = os.path.join(self._tdir,"log.csv")
        l = nodelib.log.CsvLogger(fn,'w',state=TEST_STATE_NAMES)
        for i in range(N):
            for s in TEST_STATE_NAMES:
                setattr(l,s,TEST_STATE[s](i))
            l.update()
        self.assertEqual( len(open(fn).readlines()), 1+N )

        l.close()

        i = 0
        l = nodelib.log.CsvLogger(fn,'r',state=TEST_STATE_NAMES)
        for row in l.record_iterator():
            for s in TEST_STATE_NAMES:
                self.assertEqual(TEST_STATE[s](i), TEST_STATE[s](getattr(row,s)))
            i += 1

    def test_write_threaded(self):
        N = 10000

        def _write(_l):
            for i in range(N):
                for s in TEST_STATE_NAMES:
                    setattr(_l,s,TEST_STATE[s](i))
                _l.update()

        T = 10

        l = nodelib.log.CsvLogger(self._fn,'w',state=TEST_STATE_NAMES)

        threads = []
        for _ in range(10):
            t = threading.Thread(target=_write, args=(l,))
            threads.append( t )
            t.start()

        for t in threads:
            t.join()

        fn = l.close()

        self.assertEqual( len(open(fn).readlines()), 1+(N*T) )


        #check all lines are valid when written concurrently from many
        #threads
        i = 0
        l = nodelib.log.CsvLogger(fn,'r',state=TEST_STATE_NAMES)
        for row in l.record_iterator():
            i += 1

        self.assertEqual(i, N*T)

    def tearDown(self):
        shutil.rmtree(self._tdir)

if __name__=='__main__':
    unittest.main()

