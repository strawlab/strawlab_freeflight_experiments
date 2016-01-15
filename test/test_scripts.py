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
                "--no-cached --zfilt trim --zfilt-max 0.45 --rfilt trim --rfilt-max 0.5 "\
                "--arena flycube --outdir %s --idfilt 4200" % self._tdir,
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
                "--no-cached --zfilt trim --rfilt trim --lenfilt 1 "\
                "--arena flycave --outdir %s --idfilt 9" % self._tdir,
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
                "--no-cached --lenfilt 1 --idfilt 9 --rfilt none --zfilt none --lenfilt 1 --arena flycave --save-plot "\
                "--outdir %s" % self._tdir,
                shell=True,
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                cwd=self._sdir)
        stdout,stderr = proc.communicate()
        self.assertEqual(proc.returncode,0,stderr)
        #uuid_obj_id_fn0_condition
        self.assertTrue(os.path.isfile(os.path.join(self._tdir,"trajectory-viewer.py","9_11900_checkerboard16pnginfinitysvg031000102.png")))

    def test_conflict_analysis(self):
        proc = subprocess.Popen(
                "./conflict-analysis.py --uuid 74bb2ece2f6e11e395fa6c626d3a008a "\
                "--no-cached --zfilt trim --rfilt trim --lenfilt 1 --arena flycave --outdir %s" % self._tdir,
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
                "--no-cached --zfilt trim --rfilt trim --arena flycave --outdir %s" % self._tdir,
                shell=True,
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                cwd=self._sdir,
                env=self._env)
        stdout,stderr = proc.communicate()
        self.assertEqual(proc.returncode,0,stderr)
        self.assertTrue(os.path.isfile(os.path.join(self._tdir, "confinement-analysis.py","20130530_160318.traces.png")))
        #
        #         Exception: There are overlapping trials!
        #     index                          pre_uuid  pre_oid  pre_frame0  pre_endf  \
        # 0      42  3cdbff26c93211e2b3606c626d3a008a    10915     1000024   1000828
        # 1     157  3cdbff26c93211e2b3606c626d3a008a    11270     1043593   1044430
        # 2      72  3cdbff26c93211e2b3606c626d3a008a    27524     4855208   4856063
        # 3      78  3cdbff26c93211e2b3606c626d3a008a    28226     5131749   5133150
        # 4      83  3cdbff26c93211e2b3606c626d3a008a    29011     5371520   5372872
        # 5      84  3cdbff26c93211e2b3606c626d3a008a    29011     5372022   5372872
        # 6     216  3cdbff26c93211e2b3606c626d3a008a    31299     6003984   6005638
        # 7     101  3cdbff26c93211e2b3606c626d3a008a    31705     6177407   6177970
        # 8     224  3cdbff26c93211e2b3606c626d3a008a    31722     6192419   6193066
        # 9     109  3cdbff26c93211e2b3606c626d3a008a    32261     6349298   6351173
        # 10    110  3cdbff26c93211e2b3606c626d3a008a    32261     6350303   6351173
        #
        #     index                         post_uuid  post_oid  post_frame0  post_endf
        # 0      43  3cdbff26c93211e2b3606c626d3a008a     10915      1000531    1000828
        # 1     158  3cdbff26c93211e2b3606c626d3a008a     11270      1044093    1044430
        # 2      73  3cdbff26c93211e2b3606c626d3a008a     27524      4855712    4856063
        # 3      79  3cdbff26c93211e2b3606c626d3a008a     28226      5132253    5133150
        # 4      84  3cdbff26c93211e2b3606c626d3a008a     29011      5372022    5372872
        # 5      85  3cdbff26c93211e2b3606c626d3a008a     29011      5372524    5372872
        # 6     217  3cdbff26c93211e2b3606c626d3a008a     31299      6004486    6005638
        # 7     222  3cdbff26c93211e2b3606c626d3a008a     31705      6177605    6177970
        # 8     225  3cdbff26c93211e2b3606c626d3a008a     31722      6192923    6193066
        # 9     110  3cdbff26c93211e2b3606c626d3a008a     32261      6350303    6351173
        # 10    111  3cdbff26c93211e2b3606c626d3a008a     32261      6350803    6351173

    def test_perturbation_analysis(self):
        proc = subprocess.Popen(
                "./perturbation-analysis.py --uuid 2a8386e0dd1911e3bd786c626d3a008a "\
                "--no-cached --zfilt trim --rfilt trim --lenfilt 1 --arena flycave --outdir %s" % self._tdir,
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
                "--no-cached --zfilt trim --rfilt trim --lenfilt 1 --arena flycave --outdir %s" % (
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
