#!/usr/bin/env python
import os.path
import tempfile
import subprocess
import unittest
import shutil

import roslib.packages

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
        proc = subprocess.Popen(
                "./rotation-analysis.py --uuid 0b813e6435ac11e3944b10bf48d76973 "\
                "--zfilt trim --zfilt-max 0.45 --rfilt trim --rfilt-max 0.5 "\
                "--arena flycube --reindex --outdir %s --idfilt 4200" % self._tdir,
                shell=True,
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                cwd=self._sdir,
                env=self._env)
        stdout,stderr = proc.communicate()
        self.assertEqual(proc.returncode,0,stderr)
        self.assertTrue(os.path.isfile(os.path.join(self._tdir,"rotation-analysis.py","20131015_171030.traces.png")))

    def test_rotation_analysis_flycave(self):
        proc = subprocess.Popen(
                "./rotation-analysis.py --uuid 9b97392ebb1611e2a7e46c626d3a008a "\
                "--zfilt trim --rfilt trim --lenfilt 1 --reindex "\
                "--outdir %s --idfilt 9" % self._tdir,
                shell=True,
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                cwd=self._sdir,
                env=self._env)
        stdout,stderr = proc.communicate()
        self.assertEqual(proc.returncode,0,stderr)
        self.assertTrue(os.path.isfile(os.path.join(self._tdir, "rotation-analysis.py","20130512_170802.traces.png")))

    def test_trajectory_viewer(self):
        proc = subprocess.Popen(
                "./trajectory-viewer.py --uuid 9b97392ebb1611e2a7e46c626d3a008a "\
                "--lenfilt 1 --idfilt 9 --rfilt none --zfilt none --lenfilt 1 "\
                "--outdir %s --save-data" % self._tdir,
                shell=True,
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                cwd=self._sdir)
        stdout,stderr = proc.communicate()
        self.assertEqual(proc.returncode,0,stderr)
        self.assertTrue(os.path.isfile(os.path.join(self._tdir,"9b97392ebb1611e2a7e46c626d3a008a_9_checkerboard16pnginfinitysvg0310001020.csv")))
        self.assertTrue(os.path.isfile(os.path.join(self._tdir,"checkerboard16pnginfinitysvg0310001020.png")))

    def test_conflict_analysis(self):
        proc = subprocess.Popen(
                "./conflict-analysis.py --uuid 74bb2ece2f6e11e395fa6c626d3a008a "\
                "--zfilt trim --rfilt trim --lenfilt 1 --reindex --outdir %s" % self._tdir,
                shell=True,
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                cwd=self._sdir,
                env=self._env)
        stdout,stderr = proc.communicate()
        self.assertEqual(proc.returncode,0,stderr)
        self.assertTrue(os.path.isfile(os.path.join(self._tdir, "conflict-analysis.py","20131007_183416.traces.png")))

    def test_confinement_analysis(self):
        proc = subprocess.Popen(
                "./confinement-analysis.py --uuid 3cdbff26c93211e2b3606c626d3a008a "\
                "--zfilt trim --rfilt trim --reindex --outdir %s" % self._tdir,
                shell=True,
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                cwd=self._sdir,
                env=self._env)
        stdout,stderr = proc.communicate()
        self.assertEqual(proc.returncode,0,stderr)
        self.assertTrue(os.path.isfile(os.path.join(self._tdir, "confinement-analysis.py","20130530_160318.traces.png")))

    def test_perturbation_analysis(self):
        proc = subprocess.Popen(
                "./perturbation-analysis.py --uuid 2a8386e0dd1911e3bd786c626d3a008a "\
                "--zfilt trim --rfilt trim --lenfilt 1 --arena flycave --outdir %s" % self._tdir,
                shell=True,
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                cwd=self._sdir,
                env=self._env)
        stdout,stderr = proc.communicate()
        self.assertEqual(proc.returncode,0,stderr)
        self.assertTrue(os.path.isfile(os.path.join(self._tdir, "perturbation-analysis.py","COMPLETED_PERTURBATIONS.md")))

    def test_csv_h5(self):
        ddir = os.path.join(
                    roslib.packages.get_pkg_dir('strawlab_freeflight_experiments'),
                   'test','data','0'*32
        )
        proc = subprocess.Popen(
                "./rotation-analysis.py --csv-file %s --h5-file %s "\
                "--zfilt trim --rfilt trim --lenfilt 1 --arena flycave --outdir %s" % (
                    os.path.join(ddir,'20131004_161631.rotation.csv'),
                    os.path.join(ddir,'20131004_161625.simple_flydra.h5'),
                    self._tdir),
                shell=True,
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                cwd=self._sdir,
                env=self._env)
        stdout,stderr = proc.communicate()
        self.assertEqual(proc.returncode,0,stderr)
        self.assertTrue(os.path.isfile(os.path.join(self._tdir, "rotation-analysis.py","data.json")))

    def tearDown(self):
        shutil.rmtree(self._tdir)

if __name__=='__main__':
    unittest.main()

