# coding=utf-8
"""Convenient access to files in the strawlab (IMP) data collection infrastructure."""
from ast import literal_eval
from collections import defaultdict
import json
from glob import glob
from itertools import izip, chain
import os
import os.path as op
import pickle
import shutil
import datetime
import pytz

from flydata.misc import ensure_dir
from oscail.common.config import Configurable, Configuration
from flydata.log import warning, info
import numpy as np

# FIXME: this should be user configurable via commandline/envvar/configfile
STRAWLAB_DATA_ROOT = '/mnt/strawscience/data'


###############################
# Utility functions
###############################

def _autodata_byuuid_path(data_root):
    """Where would autodata be saving trajectory&co data files?"""
    return op.join(data_root, 'auto_pipeline', 'raw_archive', 'by_uuid')


def _analysis_byuuid_path(data_root):
    """Where would the analysis scripts be saving data&images&co files by default?"""
    return op.join(data_root, 'plots')


def _check_only_one(what_name, where_name):
    """Checks that one and only one file of a kind is in a directory; if so, returns it."""
    collection = glob(op.join(where_name, what_name))
    if 1 == len(collection):
        return collection[0]
    if len(collection) > 1:
        warning(' More than one %s in %s:\n\t%r' % (what_name, where_name, collection))
    elif len(collection) < 1:
        warning(' Cannot find a single %s in %s' % (what_name, where_name))


def _verbose_copy(src, dest):
    """Copies src into dest with a couple of useful feedback messages and not failing if src does not exist."""
    if not op.isfile(src):
        warning('\tThe file %s does not exist, skipping copy...' % src)
        return
    info('\tCopying %s (%.2f MiB)' % (op.basename(src), op.getsize(src) / 1024. ** 2))
    ensure_dir(dest)
    shutil.copy(src, dest)


###############################
# Files from analysis scripts / combine framework
###############################

def _has_combined_data(dire):
    """Returns True iff dire is a directory and contains freeflight trajectories merged with the Combine framework."""
    return op.isdir(dire) and op.isfile(op.join(dire, 'data.pkl'))


def _all_combined_data_dirs(root):
    """Returns all top-level directories that contain combined data under root."""
    return [op.join(root, dire) for dire in [root] + os.listdir(root)
            if _has_combined_data(op.join(root, dire))]


def _one_combined_data_dir(root, analysis_name=None):
    """Returns an exploratory data analysis root from one of freeflight analysis scripts.

    If analysis_name is None, it returns the subdirectory with the most recent analysis
    (including root itself).

    Examples:
    >>> root = '/mnt/strawscience/data/plots/0056e1faf93511e2b4df6c626d3a008a'
    >>> print _one_combined_data_dir(root)
    /mnt/strawscience/data/plots/0056e1faf93511e2b4df6c626d3a008a/rotation-analysis.py
    >>> print _one_combined_data_dir(root, analysis_name='rotation-analysis.py')
    /mnt/strawscience/data/plots/0056e1faf93511e2b4df6c626d3a008a/rotation-analysis.py
    >>> root = '/mnt/strawscience/data/plots/fc0fae30930611e3947f10bf48d7699b'
    >>> print _one_combined_data_dir(root)
    /mnt/strawscience/data/plots/fc0fae30930611e3947f10bf48d7699b
    >>> root = '/mnt/strawscience/data/plots/fb7ba83ccbb611e3af9d6c626d3a008a'
    >>> print _one_combined_data_dir(root)
    /mnt/strawscience/data/plots/fb7ba83ccbb611e3af9d6c626d3a008a/perturbation-analysis.py
    """
    # Infer the latest analysis
    if analysis_name is None:
        dirs = _all_combined_data_dirs(root)
        if not dirs:
            raise Exception('%s does not contain analysis data' % root)
        return max(dirs, key=op.getmtime)
    # Check requested analysis
    requested_analysis = op.join(root, analysis_name)
    if not _has_combined_data(requested_analysis):
        raise Exception('%s does not contain analysis data' % requested_analysis)
    return requested_analysis


def _parse_readme(readme_file):
    """
    Parses a readme file from John's analysis scripts, extracting the exact parameters of the analysis.

    Usually these analysis correspond to a manual filtering/cleanup of the trajectories in the experiment
    plus the generation of analysis plots.

    Parameters
    ----------
    readme_file : string or file-like object
        Path to the readme file or file object

    Returns
    -------
    A tuple (command_line, options_dict), where command_line is the command line used to generate the analysis
    and options_dict contains a dictionary of all the options (customized and default) values.
    """
    if isinstance(readme_file, str):
        with open(readme_file) as reader:
            lines = reader.readlines()
    else:
        lines = readme_file.readlines()
    command_line = lines[2]
    # Some sanity checking
    if '.py' not in command_line or 'zfilt' not in command_line:
        raise Exception('Unexpected command line data in %s readme, please check' % readme_file)
    # Regenerate the options dictionary
    options_dict = {param_name.strip(): literal_eval(param_value.strip())
                    for param_name, param_value in izip(lines[5::2], lines[6::2])
                    if param_name.strip()}
    # More sanity checking
    if 'zfilt' not in options_dict:
        raise Exception('zfilt should be in the options dictionary in %s' % readme_file)
    if 'rfilt' not in options_dict:
        raise Exception('rfilt should be in the options dictionary in %s' % readme_file)
    return command_line.strip(), options_dict  # Missing maybe software version


class FreeflightAnalysisFiles(Configurable):
    """
    Provides convenient access to the files generated by the freeflight_analysis scripts.

    Examples of analysis scripts from freeflight_analysis (a non exhaustive list):
      - confinement-analysis.py
      - perturbation-analysis.py
      - rotation-analysis-zoom.py
      - translation-analysis.py
      - conflict-analysis.py
      - rotation-analysis.py

    Actually what we need from here is just the preprocessing part
    (merging and filtering stimuli and response data generated by freeflight_analysis.analysislib.combine framework).
    This object is a configurable that generates its id based on the filtering options
    (these affecting the data itself).

    Parameters
    ----------
    analysis_dir : string
        The path of the parent directory containing the analysis files
    """

    def __init__(self, analysis_dir):

        super(FreeflightAnalysisFiles, self).__init__()

        self._analysis_dir = analysis_dir

        self._analysis_json_file = op.join(self._analysis_dir, 'data.json')  # Metadata like dt, conditions,
                                                                             # long trajectories,
                                                                             # parameters for the analysis...

        self._analysis_pkl_file = op.join(self._analysis_dir, 'data.pkl')  # A pickled bunch of trajectories and
                                                                           # metadata merged with the
                                                                           # Combiner framework
                                                                           # from strawlab_freeflight

        self._analysis_readme_file = op.join(self._analysis_dir, 'README')  # Complete information on how the analysis
        # was generated

        self._analysis_notes_file = op.join(self._analysis_dir, 'NOTES')  # Manually introduced information
                                                                          # in the strawcore website

        # Lazy properties
        self._analysis_data = None
        self._analysis_json_data = None
        self._analysis_command_line = None
        self._analysis_command_options = None

    def _read_json_data(self):
        with open(self._analysis_json_file) as reader:
            return json.load(reader)

    def _read_data(self):
        with open(self._analysis_pkl_file) as reader:
            return pickle.load(reader)

    def _cached_data(self):
        if self._analysis_data is None:
            self._analysis_data = self._read_data()
        return self._analysis_data  # TODO: weakref this if needed

    def _cached_json_data(self):
        if self._analysis_json_data is None:
            self._analysis_json_data = self._read_json_data()
        return self._analysis_json_data

    def pkl_modification_datetime(self, tzinfo=pytz.timezone('Europe/Vienna')):
        return datetime.datetime.fromtimestamp(op.getmtime(self._analysis_pkl_file), tz=tzinfo)

    def analysis_command_line(self):
        if self._analysis_command_line is None:
            self._analysis_command_line, self._analysis_command_options = _parse_readme(self._analysis_readme_file)
        return self._analysis_command_line.strip()

    def analysis_command_options(self):
        if self._analysis_command_options is None:
            self._analysis_command_line, self._analysis_command_options = _parse_readme(self._analysis_readme_file)
        return self._analysis_command_options

    def _options_subset(self, *option_names):
        """Returns a list of (option_name, value) for the specified option names.
        If the option is not present in the README file, we assume a value of None.
        """
        # FIXME: do not assume None, it is not correct, for example, for frames_before; defaults should be sensible
        co = self.analysis_command_options()
        return [(option_name, co.get(option_name, None)) for
                option_name in option_names]

    def filtering_zfilt_options(self):
        #
        # filtering by z component - flies too high or too low are highly likely to be landing / out of VR
        #
        return self._options_subset('zfilt', 'zfilt_max', 'zfilt_min')

    def filtering_rfilt_options(self):
        #
        # filtering by x,y radius- flies too far from the center are highly likely to be landing / out of VR
        #
        return self._options_subset('rfilt', 'rfilt_max')

    def filtering_customfilt_options(self):
        #
        # Custom-filtering allows to inject a python-program string that will be evaluated in a context
        # with one variable of interest: "df", the data-frame of time series.
        # N.B. code injection and, worse, hard to standardize
        # (even if we do not allow spaces in the string, we could for example provide two equivalent strings
        # like df[df['velocity']>0.1] and df[0.1<df['velocity']]")
        # Example: if we provide
        #    "df[df['velocity']>0.1]"
        #    Then we are only keeping parts of the trajectories with velocity bigger than 0.1
        #    This can potentially screw-up trajectories
        #    (e.g. if applied before the z and radious filters,
        #    a fly might land and take off and in the landed place there will be a hole)
        #
        return self._options_subset('customfilt', 'customfilt_len')

    def filtering_lenfilt_options(self):
        #
        # Simply filtering out trajectories that are shorter than x seconds
        #
        return self._options_subset('lenfilt')

    def filtering_tfilt_options(self):
        #
        # Note that only one of these should be provided at a time?
        #
        return self._options_subset('tfilt_before', 'tfilt_after')

    def merging_frames_before_options(self):
        #
        # We would need to rerun all the analyses so that frames_before appear as an option
        # N.B. at the moment it can appear to be 0 when it actually is not, because it just do not appear
        #      unfortunately we cannot add a fix by looking at the metadata, because duration is not specified there
        # N.B. we should actually know of the default value = 0, do not make it None; same for other parameters
        #
        return self._options_subset('frames_before')

    def filtering_idfilt_options(self):
        #
        # Maybe this could be put in subsequent analyses, instead of here
        # (because it is object selection, subsets of the data can be recreated
        # for free after all the other cleanship is done...)
        #
        return self._options_subset('idfilt')

    def merging_filtering_options(self):
        return list(chain(
            # Add "junk frames" before the trial actually starts? we do need to know about this at any time
            self.merging_frames_before_options(),
            # Cleaning up trajectories by geometrical constraints indicating valid arena locations
            # ALways using "trim" (cut trajectories as soon as there is a violation)
            self.filtering_zfilt_options(),
            self.filtering_rfilt_options(),
            # This is dangerous and we should maybe minimize its use
            # (create functions/classes for specific useful cases)
            # Also it can break assumptions like that there are no holes in the trajectories
            # (in that case we would need to keep the index and take into account in subsequent analysis)
            self.filtering_customfilt_options(),
            # These three select concrete trajectories based on their length, datetime and id
            # Maybe it could be good to defer this to analysis (as done by the discriminative-analysis code)
            self.filtering_lenfilt_options(),
            self.filtering_tfilt_options(),
            self.filtering_idfilt_options()))

    def configuration(self):
        return Configuration(
            self.__class__.__name__,
            configuration_dict=dict(self.merging_filtering_options()))

    def dt(self):
        """Returns the sampling period in seconds."""
        return self._cached_json_data()['dt']

    def sampling_period(self):
        """Returns the sampling period in seconds."""
        return self.dt()

    def conditions(self):
        """Returns a sorted list of the conditions (strings) present in the experiments."""
        return sorted(self._cached_json_data()['conditions'])

    def trajs(self, conditions=None):
        """Returns an iterator of tuples (condition, x0, y0, oid, framenumber0, time0, condition, df)
        for the given conditions.

        Here:
          - condition is the condition string
          - x0, y0 are the coordinates of the first tracked position during the trial
            they might not correspond to the beginning of the df if frames_before is > 0
          - oid is the object id of the trajectory
            together with framenumber0 is a unique identifier for the trial
            there can be several combinations of (oid, framenumber0) for the same oid
          - framenumber0 is the starting frame number of the *trial*
          - time0 is the timestamp of the first measurement of the *trial*
          - df is a pandas dataframe containing the time series
            (x, y, z, and possibly stimulus and derived quantities series)
            It can be 0-based-indexed (starting from zero) or timestamp-indexed.

        Parameters
        ----------
        conditions : string list or string, default None
            The condition ids for which we want the trajectories.
            If None, then trajectories for all conditions will be returned.
            If a string or a list of strings, all the trajectories for the specified conditions will be returned.
        """
        if conditions is None:
            conditions = self.conditions()
        elif isinstance(conditions, basestring):
            conditions = [conditions]
        starts = []
        dfs = []
        iconditions = []
        for condition in conditions:
            if condition not in self.conditions():
                raise Exception('Condition %s not found in data from %s.'
                                'Known conditions are: \n\t%s' %
                                (condition, self._analysis_dir, '\n\t'.join(self.conditions())))
            starts.append(self._cached_data()['results'][condition]['start_obj_ids'])
            dfs.append(self._cached_data()['results'][condition]['df'])
            iconditions.extend([condition] * len(starts[-1]))
            if len(starts[-1]) != len(dfs[-1]):
                raise Exception('Inconsistency reading data from %s, condition %s.'
                                'There should be the same number of "starts" and "dataframes"' %
                                (self._analysis_dir, condition))
        return ((condition, x0, y0, obj_id, framenumber0, time0, df) for
                condition, (x0, y0, obj_id, framenumber0, time0), df in
                izip(iconditions, chain(*starts), chain(*dfs)))  # generators are useless when all is in mem already

    def mirror_to(self,
                  dest_dir,
                  mirror_analysis_data=True,
                  mirror_analysis_plots=False):

        # Analysis data
        if mirror_analysis_data:
            info(' Mirroring %s analysis data to %s' % (self._analysis_dir, dest_dir))
            _verbose_copy(self._analysis_json_file, dest_dir)
            _verbose_copy(self._analysis_pkl_file, dest_dir)
            _verbose_copy(self._analysis_readme_file, dest_dir)

        # Analysis plots
        if mirror_analysis_plots:
            info(' Mirroring %s analysis plots to %s' % (self._analysis_dir, dest_dir))
            for plot_file in glob(op.join(self._analysis_dir, '*.png')):
                _verbose_copy(plot_file, dest_dir)

    def objects_with_nans(self, variables=('x', 'y', 'z')):
        """Returns all the object-ids which contain NaN in some (basic) quantities."""
        # It should return condition-oid
        # This is bogus if we do not remove oid-ambiguities
        return [oid for _, _, oid, _, _, df in self.trajs() if df[variables].isnull().sum() > 0]

    @staticmethod
    def from_uuid(uuid, filter_id=None):
        return FreeflightAnalysisFiles(
            _one_combined_data_dir(op.join(_analysis_byuuid_path(STRAWLAB_DATA_ROOT), uuid), filter_id))


if __name__ == '__main__':

    def check_problems(uuid):
        print 'UUID=%s' % uuid
        path = op.join('/mnt/strawscience/data/plots/%s' % uuid)
        analyses = _all_combined_data_dirs(path)
        print 'There are %d analysis' % len(analyses)
        for analysis in analyses:
            print '\t%s' % analysis
            ffa = FreeflightAnalysisFiles(analysis)
            print ffa.configuration().id()
            for condition in ffa.conditions():
                oid2data = defaultdict(list)
                for x0, y0, oid, framenumber0, time0, df in ffa.trajs(condition):
                    # Consistency of x0, y0
                    if np.isnan(x0) or np.isnan(y0):
                        print 'Data %d at %s, condition %s, contains NaNs' % (oid, analysis, condition)
                    else:
                        assert (x0, y0) == (df['x'].values[0], df['y'].values[0]), \
                            '(%g, %g) != (%g, %g)' % (x0, y0, df['x'].values[0], df['y'].values[0])
                    # Uniqueness of obj_id in the condition
                    oid2data[oid].append((framenumber0, time0, df))
                for oid, data in oid2data.iteritems():
                    if len(data) > 1:
                        print 'Repeated oid=%d, trajs=%d' % (oid, len(data))
                        for framenumber0, time0, df in data:
                            print framenumber0, len(df), framenumber0 + len(df)

    UUIDs = [
        '0056e1faf93511e2b4df6c626d3a008a',
        '127f617ebf3511e3a69810bf48d76973',
        '17ff102eba7e11e385af10bf48d76973',
        '185607a2bc0f11e3a8d110bf48d76973',
        '5827179ed92511e3b15910bf48d76973',
        '62fc1158d53611e3b09710bf48d76973',
        '6b7f3facd85c11e3a99310bf48d76973',
        '769e6cf8dab611e3b64f10bf48d76973',
        '77023d5eb9b511e3b02210bf48d76973',
        '8ce92230d9eb11e3ad5010bf48d76973',
        '900bfc06bb4811e3a14f10bf48d76973',
        'ca9efe66d46d11e3b78110bf48d76973',
        'd3d1b92cbe6c11e3b54f10bf48d76973',
        'fb7ba83ccbb611e3af9d6c626d3a008a',
        'fc0fae30930611e3947f10bf48d7699b',
    ]
    for uuid in UUIDs:
        check_problems(uuid)


#######################
#
# TODO: conciliate with autodata.files, make sure we always sync with data collection
#
# TODO: look at flycave/scripts/copy-exp-data-by-uuid
#
# TODO: allow individual csv + simple_flydra files (trivial)
#
######################