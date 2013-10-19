#!/usr/bin/env python
import os.path
import tempfile
import subprocess
import unittest
import shutil

class TestScript(unittest.TestCase):

    def setUp(self):
        self._tdir = tempfile.mkdtemp()
        self._sdir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','scripts'))

        #remove DISPLAY, and dont write svg nor pickle
        self._env = os.environ.copy()
        try:
            del self._env['DISPLAY']
        except KeyError:
            pass

        self._env['WRITE_SVG']='0'
        self._env['WRITE_PKL']='0'

    def test_rotation_analysis_flycube(self):
        ret = subprocess.call(
                "./rotation-analysis.py --uuid 0b813e6435ac11e3944b10bf48d76973 "\
                "--zfilt trim --zfilt-max 0.45 --rfilt trim --rfilt-max 0.5 "\
                "--arena flycube --reindex --outdir %s --idfilt 4200" % self._tdir,
                shell=True,
                cwd=self._sdir)
        self.assertEqual(ret,0)
        self.assertTrue(os.path.isfile(os.path.join(self._tdir, "20131015_171030.traces.png")))

    def test_rotation_analysis_flycave(self):
        ret = subprocess.call(
                "./rotation-analysis.py --uuid 9b97392ebb1611e2a7e46c626d3a008a "\
                "--zfilt trim --rfilt trim --lenfilt 1 --reindex "\
                "--outdir %s --idfilt 9" % self._tdir,
                shell=True,
                cwd=self._sdir)
        self.assertEqual(ret,0)
        self.assertTrue(os.path.isfile(os.path.join(self._tdir, "20130512_170802.traces.png")))

    def test_trajectory_viewer(self):
        ret = subprocess.call(
                "./trajectory-viewer.py --uuid 9b97392ebb1611e2a7e46c626d3a008a "\
                "--lenfilt 1 --idfilt 9 --rfilt none --zfilt none --lenfilt 1 "\
                "--outdir %s --save" % self._tdir,
                shell=True,
                cwd=self._sdir)
        self.assertEqual(ret,0)
        self.assertTrue(os.path.isfile(os.path.join(self._tdir, "20130512_170802.infinity.png")))

    def test_conflict_analysis(self):
        ret = subprocess.call(
                "./conflict-analysis.py --uuid 74bb2ece2f6e11e395fa6c626d3a008a "\
                "--zfilt trim --rfilt trim --lenfilt 1 --reindex --outdir %s" % self._tdir,
                shell=True,
                cwd=self._sdir)
        self.assertEqual(ret,0)
        self.assertTrue(os.path.isfile(os.path.join(self._tdir, "20131007_183416.traces.png")))

    def test_confinement_analysis(self):
        ret = subprocess.call(
                "./confinement-analysis.py --uuid 3cdbff26c93211e2b3606c626d3a008a "\
                "--zfilt trim --rfilt trim --reindex --outdir %s" % self._tdir,
                shell=True,
                cwd=self._sdir)
        self.assertEqual(ret,0)
        self.assertTrue(os.path.isfile(os.path.join(self._tdir, "20130530_160318.traces.png")))

    def tearDown(self):
        shutil.rmtree(self._tdir)

if __name__=='__main__':
    unittest.main()

