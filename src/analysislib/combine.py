from functools import partial
from itertools import izip
import os.path
import sys
import random
import time
import cPickle
import pickle
import re
import operator
import hashlib
import datetime
import calendar
import collections
import copy
import itertools

import tables
import pandas as pd
from pandas.tseries.index import DatetimeIndex
import numpy as np
import pytz
import scipy.io
import yaml

from strawlab.constants import maybe_fake_ros, ensure_dir

maybe_fake_ros('strawlab_freeflight_experiments')

import autodata.files

import analysislib.fixes
import analysislib.args
import analysislib.features as afeat

from ros_flydra.constants import IMPOSSIBLE_OBJ_ID, IMPOSSIBLE_OBJ_ID_ZERO_POSE
from strawlab.constants import (DATE_FMT, AUTO_DATA_MNT, find_experiment,
                                uuids_from_flydra_h5, uuids_from_experiment_csv,
                                set_permissions)

from strawlab_freeflight_experiments.conditions import Condition, Conditions, ConditionCompat

from whatami import What

# results = {
#   condition:{
#       df:[dataframe,...],
#       start_obj_ids:[(x0,y0,obj_id,framenumber0,time0),...],
#       count:n_frames,
#       uuids:[uuid,...],
#   }
# }


def safe_condition_string(s):
    return "".join([c for c in s if re.match(r'\w', c)])


def condition_switches_from_controller_csv(csv):
    if not isinstance(csv, pd.DataFrame):
        csv = pd.read_csv(csv)
    cond_or_trial_change = csv[csv['lock_object'].isin((IMPOSSIBLE_OBJ_ID, IMPOSSIBLE_OBJ_ID_ZERO_POSE))]
    cond_changes = cond_or_trial_change['condition'].shift() != cond_or_trial_change['condition']
    return cond_or_trial_change[cond_changes][['condition', 't_sec', 't_nsec']]
    # could also save framenumbers from last / next obs?


def is_testing():
    """Returns True iff we detect we are testing.
    If this is the case, combine caches are never read.
    """
    return 'NOSETEST_FLAG' in os.environ or 'nosetests' in sys.argv[0]


def find_some_overlaps(expdf, start='start', end='end', as_tidy=False):
    """
    Checks that no unit overlaps any other unit in time.

    N.B. This should probably be moved to our "intervals" lib.

    Parameters
    ----------
    expdf : pandas DataFrame
      Each row should correspond to a unit (e.g. an experiment).
      The index should identify a unit.
      It should contain at least two columns: start and end.

    start : string, default 'frame0'
      The column name where the start of each unit is stored.

    end : string, default 'end'
      The column name where the end of each unit is stored.

    Returns
    -------
    A list of row tuples (index1, index2) indicating overlapping units.
    (index1, index2) are the corresponding index values for the overlapping rows.
    Albeit possibly non exhaustive, if there exist an overlap, this list is guaranteed to be non-empty.
    Returns an empty list iff there are no two overlapping units.

    Examples
    --------
    Here we find the two overlapping units
    >>> df = pd.DataFrame([[0, 3], [1, 10], [9, 12]], columns=['start', 'end'])
    >>> find_some_overlaps(df, start='start', end='end')
    [(0, 1), (1, 2)]

    But the returned overlapping intervals are not exhaustive (note (0, 2) is missing from the result).
    >>> df = pd.DataFrame([[0, 30], [5, 10], [10, 12]], columns=['start', 'end'])
    >>> find_some_overlaps(df, start='start', end='end')
    [(0, 1)]

    Intervals are assumed to be half-closed [start, end)
    >>> df = pd.DataFrame([[0, 3], [5, 10], [10, 12]], columns=['start', 'end'])
    >>> find_some_overlaps(df, start='start', end='end')
    []

    For empty dataframes, we return an empty list
    >>> df = pd.DataFrame([], columns=['start', 'end'])
    >>> find_some_overlaps(df, start='start', end='end')
    []

    0-length intervals and ill-defined intervals trigger a ValueError
    >>> df = pd.DataFrame([[0, 0], [0, 10]], columns=['start', 'end'])
    >>> find_some_overlaps(df, start='start', end='end')
    Traceback (most recent call last):
     ...
    ValueError: Intersections might be ill defined because there are empty intervals (start >= end)
    >>> df = pd.DataFrame([[0, 0], [3, 2]], columns=['start', 'end'])
    >>> find_some_overlaps(df, start='start', end='end')
    Traceback (most recent call last):
     ...
    ValueError: Intersections might be ill defined because there are empty intervals (start >= end)
    """
    if (expdf[end] <= expdf[start]).any():
        raise ValueError('Intersections might be ill defined because there are empty intervals (start >= end)')
    try:
        expdf = expdf.sort_values(by=[start, end])  # pandas >= 0.17
    except AttributeError:
        expdf = expdf.sort([start, end])  # deprecated
    starts_before_previous_ends = expdf[start].shift(-1).iloc[:-1] < expdf[end].iloc[:-1]
    if starts_before_previous_ends.any():
        firsts = expdf.iloc[:-1][starts_before_previous_ends.values].index
        overlaps = expdf.iloc[1:][starts_before_previous_ends.values].index
        return list(zip(firsts, overlaps))
    return []


def overlaps2tidy(df, overlaps, columns=('uuid', 'oid', 'startf')):

    def loc_and_select(rows, prefix='pre_'):
        return df.loc[rows][list(columns)].rename(columns=lambda col: prefix + col).reset_index()

    if len(overlaps) > 0:
        overlaps = np.array(overlaps)
        overlaps_pre = loc_and_select(overlaps[:, 0], prefix='pre_')
        overlaps_post = loc_and_select(overlaps[:, 1], prefix='post_')
        return pd.concat([overlaps_pre, overlaps_post], axis=1)

    return None


def check_combine_health(combine, min_length_f=100):
    """Checks some invariants in combine.

    Each of these (should) have a "contract" class if flydata (or whatever we end up calling that package).
    """
    df = combine.get_trials_dataframe(startf_name='frame0', endf_name='endf')
    df = df.sort('frame0')
    check_trials_health(df, dt=combine.dt, min_length_f=min_length_f, start='frame0', end='endf')


def check_trials_health(df, dt=0.01, min_length_f=100, start='startf', end='endf'):

    #
    # See also flycave/scripts/filemaintenance/mainbrains.py for checks on similar issues
    # (holes, jumps in time...) for mainbrain files.
    #
    # These are more composable and it would repay to bring together these two realisations
    # of data contracts dynamic checks.
    #

    # Check all trajectories are long enough; we never allow 0-length trajectories
    if min_length_f is None or min_length_f < 1:
        min_length_f = 1
    # noinspection PyUnresolvedReferences
    if (df.length_f < min_length_f).any():
        raise Exception('There are trials without observations')

    # Check there are not overlapping trajectories
    overlapping = []
    for uuid, expdf in df.groupby('uuid'):
        overlaps = find_some_overlaps(expdf, start=start, end=end)
        if len(overlaps) > 0:
            overlapping.append(overlaps2tidy(expdf, overlaps, columns=['uuid', 'oid', start, end]))
    if len(overlapping) > 0:
        raise Exception('There are overlapping trials!\n%s' % str(pd.concat(overlapping, ignore_index=True)))

    # Check that trial id is indeed unique
    if not len(df.groupby(['uuid', 'oid', start])) == len(df):
        raise Exception('There are duplicated (uuid, oid, %s) tuples!' % start)

    # Check that there are no holes in the dataframes
    def has_holes(df, dt):
        observations_distances = df.index.values[1:] - df.index.values[0:-1]
        if isinstance(df.index, DatetimeIndex):
            return (observations_distances != dt).any()
        return (observations_distances != 1).any()

    with_holes = df[df['series'].apply(partial(has_holes, dt=dt))]
    if len(with_holes) > 0:
        raise Exception('There are trajectories with holes: \n%s' %
                        with_holes[['uuid', 'oid', start]].to_string())

    # Check no missings in x, y, z
    def has_missings(df, cols=('x', 'y', 'z')):
        return 0 != np.count_nonzero(df[list(cols)].isnull())
    with_missing = df[df['series'].apply(has_missings)]
    if len(with_missing) > 0:
        raise Exception('There are trajectories with unexpected missing values: \n%s' %
                        with_missing[['uuid', 'oid', start]].to_string())


def load_conditions(uuid):
    """
    Returns a dictionary of condition names and slash separated
    strings after applying any fixes
    """
    fm = autodata.files.FileModel(uuid=uuid)
    csv_fname = fm.get_file_model('*.csv').fullpath
    h5_fname = fm.get_file_model('simple_flydra.h5').fullpath

    this_exp_metadata = None
    try:
        metadata_fname = fm.get_file_model('experiment.yaml').fullpath
        with open(metadata_fname, 'r') as f:
            this_exp_metadata = yaml.safe_load(f)
    except autodata.files.FileModelException:
        pass

    if this_exp_metadata is None:
        this_exp_metadata = {}

    this_exp_metadata['csv_file'] = csv_fname
    this_exp_metadata['h5_file'] = h5_fname

    condition_fname = fm.get_file_model('condition.yaml').fullpath
    with open(condition_fname, 'r') as f:
        this_exp_metadata['conditions'] = yaml.safe_load(f)
        f.seek(0)
        condition_object = Conditions(f)

    fix = analysislib.fixes.load_csv_fixups(**this_exp_metadata)

    conditions = {}
    for cond_name in condition_object:
        cond_str = condition_object[cond_name].to_slash_separated()
        if fix.should_fix_condition_name:
            cond_name = fix.fix_condition_name(cond_name)
        if fix.should_fix_condition:
            cond_str = fix.fix_condition(cond_str)
        conditions[cond_name] = cond_str
    return conditions


class CacheError(Exception):
    pass

class _Combine(object):

    DEFAULT_FEATURES = ('vx','vy','vz','ax','ay','az','velocity','dtheta','radius','err_pos_stddev_m')

    def __init__(self, **kwargs):
        self._enable_debug = kwargs.get("debug",True)
        self._enable_warn = kwargs.get("warn",True)
        self._dt = None
        self._lenfilt = None
        self._idfilt = []
        self._skipped = {}
        self._results = {}
        self._tzname = 'Europe/Vienna'
        self._tfilt_before = None
        self._tfilt_after = None
        self._tfilt = None
        self._plotdir = None
        self._plotdir_uuids = []
        self._analysistype = None
        self._index = 'framenumber'
        self._warn_cache = {}
        self._debug_cache = {}
        self._conditions = {}
        self._condition_names = {}
        self._metadata = []
        self._condition_switches = {}  # {uuid: df['condition', 't_sec', 't_nsec']}; useful esp. when randomising
        self._configdict = {'v': 18,  # bump this version when you change delicate combine machinery
                            'index': self._index
        }

        self.features = afeat.MultiFeatureComputer(*kwargs.get("features",self.DEFAULT_FEATURES))
        self._configdict['features'] = self.features    #is whatable

    def add_feature(self, feature_name=None, column_name=None):
        if feature_name and column_name:
            raise ValueError('Only one of feature_name and column_name may be provided')
        elif feature_name:
            self.features.add_feature(feature_name)
        elif column_name:
            self.features.add_feature_by_column_added(column_name)
        else:
            raise ValueError('feature_name or column_name are required')
        self._configdict['features'] = self.features

    def set_features(self, *features):
        self.features.set_features(*features)
        self._configdict['features'] = self.features

    def get_features(self):
        return self.features.get_columns_added()

    def set_index(self, index):
        VALID_INDEXES = ('framenumber', 'none')
        if (index not in VALID_INDEXES) and (not index.startswith('time')):
            raise ValueError('index must be one of %s, time+NN (where NN is a pandas resample specifier)' %
                             ', '.join(VALID_INDEXES))
        self._index = index
        self._configdict['index'] = self._index

    def what(self):
        return What(self.__class__.__name__, self._configdict)

    def _args_to_configuration(self, args):
        for k,v in args._get_kwargs():
            if k not in analysislib.args.DATA_MODIFYING_ARGS:
                continue
            if k in ('uuid','idfilt') and v is not None and len(v):
                self._configdict[k] = sorted(v)
            else:
                self._configdict[k] = v

    def _get_cache_name_and_config_string(self):
        """Returns a two tuple (pkl, whatid).
          - pkl is the cached pkl path in the central data repository.
          - whatid is the configuration string identifying this combine object.
        """
        whatid = self.what().id()
        whatid_hash = hashlib.sha224(whatid).hexdigest()
        return os.path.join(AUTO_DATA_MNT, 'cached', 'combine', whatid_hash[:2], whatid_hash + '.pkl'), whatid

    def get_cache_pkl_path(self):
        """Returns the path that would correspond to a pickle caching this combine object in the central repository."""
        return self._get_cache_name_and_config_string()[0]

    def _read_cache_file(self):
        if is_testing():
            return None

        pkl = self.get_cache_pkl_path()
        if os.path.exists(pkl):
            self._debug("IO:     reading %s" % pkl)
            with open(pkl, "rb") as f:
                # Speed optimisation, see:
                #   http://stackoverflow.com/questions/16833124/pickle-faster-than-cpickle-with-numeric-data
                # and
                #   http://stackoverflow.com/questions/19807790/
                #   given-a-pickle-dump-in-python-how-to-i-determine-the-used-protocol
                def unpickle_fast():
                    import pickletools
                    with open(pkl, 'r') as reader:
                        if next(pickletools.genops(reader))[0].proto >= 2:
                            return cPickle.load(f)
                        return pickle.load(f)
                try:
                    return unpickle_fast()
                except:
                    try:
                        # This is inefficient, as it tries first to use cpickle again and only
                        # fallsback to compat pickle on failing.
                        # However, it is the recommended pandas way of keeping backwards compat
                        #   http://pandas.pydata.org/pandas-docs/stable/io.html#io-pickle
                        # So let's treat it as a black box and do not directly use pandas compat pickle.
                        return pd.read_pickle(pkl)
                    except Exception as e:
                        self._warn('Could not unpickle %s, recombining and recaching' % pkl)
                        self._warn('The error was %s' % str(e))
                        raise CacheError(pkl)
        return None

    def _update_plot_dir(self, args, uuid, uuids=None):
        def add_uuid(_uuid):
            if _uuid and (_uuid not in self._plotdir_uuids):
                self._plotdir_uuids.append(_uuid)

        add_uuid(uuid)
        if uuids is not None:
            for uuid in uuids:
                add_uuid(uuid)

        uuids = self._plotdir_uuids

        if args.outdir:
            self.plotdir = args.outdir
            return

        if not uuids:
            self.plotdir = os.getcwd()
        elif len(uuids) > 1:
            self.plotdir = autodata.files.get_multi_plot_path(*uuids)
        else:
            fm = autodata.files.FileModel(basedir=args.basedir)
            fm.select_uuid(uuids[0])
            self.plotdir = fm.get_plot_dir()

    def get_data_dictionary(self):
        """Returns a dictionary with the data that is worth to save from this combine object.

        These are the data we deem worthy:
          - dt and results (see get_results)
          - skipped: a dictionary {condition -> number of skipped trials}
          - conditions: a dictionary {condition -> condition_configuration_dict}
          - condition_names: a dictionary {condition -> normalised_condition_name}
          - condition_switches: a dictionary {uuid -> df}; df contains [condition, t_sec, t_nsec]
          - metadata: a list of dictionaries, each containing the metadata for one experiment
          - csv_file: the path to the original experiment csv file or None if it was not used
        """
        return {
            "results": self._results,
            "dt": self.dt,
            "skipped": self._skipped,
            "conditions": self._conditions,
            "condition_names": self._condition_names,
            "condition_switches": self._condition_switches,
            "metadata": self._metadata,
            "csv_file": self.csv_file if hasattr(self, 'csv_file') else None  # do we use CombineH5 for something?
        }

    def is_populated(self):
        return len(self._results) > 0

    def save_cache(self, pkl=None):
        central_pkl, whatid = self._get_cache_name_and_config_string()
        pkl = central_pkl if pkl is None else pkl
        # Save the pickle
        with open(pkl, "w+b") as f:
            self._debug("IO:     writing %s" % pkl)
            cPickle.dump(self.get_data_dictionary(), f, protocol=pickle.HIGHEST_PROTOCOL)
        # Save the id
        with open(os.path.splitext(pkl)[0] + '.txt', "w") as f:
            f.write(whatid)

    def symlink_central_cache_to(self, to_dir, prefix='data', overwrite=False):
        """Symlinks the central cache for this combine object into a directory.

        If the central cache does not exist, saves it.
        Note that at the moment this is brittle and would lead to "empty" cache files
        if we call this before populating the Combine object with data. So use this
        function with caution.

        Parameters
        ----------
        to_dir : string
          The directory where the cache files will be symlinked

        prefix : string, default 'data'
          The name prefix for the symlink files

        overwrite : boolean, default False
          Existing targets will be overwritten only if this is True
          Note that the cache can still be recreated even if the targets
          already exist but the cache have been purged.
        """
        # Create cache if needed
        central_cache_pkl = self.get_cache_pkl_path()
        central_cache_txt = os.path.splitext(central_cache_pkl)[0] + '.txt'
        if not os.path.isfile(central_cache_pkl) or not os.path.isfile(central_cache_txt):
            if not self.is_populated():
                raise Exception('Cannot symlink uncached, unpopulated combine')
            self.save_cache(pkl=None)
        # Symlink
        ensure_dir(to_dir)
        dest_pkl = os.path.join(to_dir, prefix + '.pkl')
        dest_txt = os.path.join(to_dir, prefix + '.txt')
        for path in (dest_pkl, dest_txt):
            if os.path.exists(path):
                if not overwrite:
                    raise IOError('The file %s already exists and overwrite is False' % path)
                os.unlink(path)
        os.symlink(central_cache_pkl, dest_pkl)
        os.symlink(central_cache_txt, dest_txt)

    def _get_trajectories(self, h5):
        trajectories = h5.root.trajectories

        #unexplainable protip - adding an index on the framenumber table makes
        #things slloooowwww
        if trajectories.cols.framenumber.is_indexed:
            trajectories.cols.framenumber.removeIndex()

        if not trajectories.cols.obj_id.is_indexed:
            try:
                trajectories.cols.obj_id.createIndex()
            except tables.exceptions.FileModeError:
                self._warn("obj_id column not indexed, this will be slow. reindex")

        return trajectories

    def _maybe_apply_tfilt_should_save(self, ts):
        #returns true to indicate the data should be saved
        if (self._tfilt_before is None) and (self._tfilt_after is None):
            return True

        #now we should have parsed an h5 file, so can actually create a
        #timestamp in utc from the timezone of the data and the datetime spec
        #as given
        if self._tfilt is None:
            spec = self._tfilt_before if self._tfilt_before is not None else self._tfilt_after
            lt = self.timezone.localize(spec)
            self._tfilt = calendar.timegm(lt.utctimetuple())

        if self._tfilt_before is not None:
            return ts < self._tfilt
        else:
            return ts > self._tfilt

    def _maybe_add_tfilt(self, args):
        for f in ("tfilt_before", "tfilt_after"):
            v = getattr(args,f,None)
            if v is not None:
                setattr(self, "_%s" % f, datetime.datetime.strptime(v, DATE_FMT))

    def _debug(self, m):
        if self._enable_debug:
            print m

    def _debug_once(self, m):
        if m not in self._debug_cache:
            self._debug(m)
            self._debug_cache[m] = 0
        self._debug_cache[m] += 1

    def _warn(self, m):
        if self._enable_warn:
            print m

    def _warn_once(self, m):
        if m not in self._warn_cache:
            self._warn(m)
            self._warn_cache[m] = 0
        self._warn_cache[m] += 1

    def _get_df_sample_interval(self, df=None):
        try:
            if df is not None:
                return df.index.freq.nanos / 1e9
            else:
                #take the first dataframe as all must have the same index
                for cond,r in self._results.iteritems():
                    for df in r['df']:
                        return df.index.freq.nanos / 1e9
        except AttributeError:
            #not datetime index
            pass
        return None

    def __repr__(self):
        if not self._results:
            return "<Combine NO_DATA>"
        else:
           return "<Combine %s idx=%s dt=%s>" % (os.path.basename(self.csv_file),self._index,self.dt)

    @property
    def cache_version(self):
        return self._configdict['v']

    @property
    def dt(self):
        if not self._results:
            raise ValueError("instance contains no data")

        dt = self._get_df_sample_interval()
        if dt is None:
            #framenumber index, or original index. dt determined by
            #tracking framerate
            dt = self._dt

        return dt

    @property
    def has_time_index(self):
        try:
            for cond,r in self._results.iteritems():
                for df in r['df']:
                    #present only on time indexes
                    df.index.freq.nanos
                    return True
        except AttributeError:
            pass
        return False

    @property
    def analysis_type(self):
        if self._analysistype is None:
            return os.path.basename(sys.argv[0])
        else:
            return self._analysistype

    @analysis_type.setter
    def analysis_type(self, val):
        self._analysistype = val

    @property
    def plotdir(self):
        """the directory in which to store plots for this analysis"""
        if self._plotdir is None:
            #none before a csv file has been added / args have been processed
            #(because of --outdir)
            return self._plotdir

        pd = os.path.join(self._plotdir, self.analysis_type)
 
        if not os.path.isdir(pd):
            os.makedirs(pd)

        if len(self._plotdir_uuids) > 1:
            with open(os.path.join(self._plotdir,'UUIDS'),'w') as f:
                f.write('\n'.join(self._plotdir_uuids))
                f.write('\n')

        return pd

    @plotdir.setter
    def plotdir(self, val):
        self._plotdir = val

    @property
    def timezone(self):
        """the timezone in which the experiment was started"""
        return pytz.timezone(self._tzname)

    @property
    def fname(self):
        return self.get_plot_filename(
                        os.path.basename(self.csv_file).split('.')[0])

    @property
    def min_num_frames(self):
        """the minimum number of frames for the given lenfilt and dt"""
        try:
            return self._lenfilt / self._dt
        except TypeError:
            return 1

    @property
    def framerate(self):
        """the framerate of the data"""
        return 1.0 / self._dt

    def get_plot_filename(self, name, subdir=''):
        """return a full path to the autodata directory to save any plots"""
        if subdir:
            pd = os.path.join(self.plotdir,subdir)
            if not os.path.isdir(pd):
                os.makedirs(pd)
        return os.path.join(self.plotdir,subdir,name)

    def get_num_skipped(self, condition):
        """returns the number of skipped trials for the given condition.

        trials are skipped if they do not meet all the filter criteria"""
        return self._skipped.get(condition,0)

    def get_total_skipped(self):
        """returns the total trials skipped for all conditions"""
        return sum([self.get_num_skipped(c) for c in self.get_conditions()])

    def get_num_analysed(self, condition):
        """returns the number of trials which met the filter criteria"""
        return self._results[condition]['count']

    def get_total_analysed(self):
        """returns the total number of trials which met the filter criteria for all conditions"""
        return sum([self.get_num_analysed(c) for c in self.get_conditions()])

    def get_num_trials(self, condition):
        """returns the total number of trials in the data for the givent condition"""
        return self.get_num_skipped(condition) + self.get_num_analysed(condition)

    def get_total_trials(self):
        """returns the total trials for all conditions"""
        return sum([self.get_num_trials(c) for c in self.get_conditions()])

    def get_num_conditions(self):
        """returns the number of experimental conditions"""
        return len(self._results)

    def get_conditions(self):
        """returns a list of the names of the experimental conditions"""
        return self._results.keys()

    def get_condition_name(self, cond):
        """return a printable human readable condition name"""
        return self._condition_names.get(cond,cond)

    def get_condition_configuration(self, cond):
        """returns the full dictionary that defines the experimental condition"""
        return self._conditions.get(cond, {})

    def get_condition_object(self, cond):

        name = self.get_condition_name(cond)
        condition_conf = self.get_condition_configuration(cond)

        if condition_conf:
            obj = Condition(condition_conf)
            obj.name = name
            return obj
        else:
            #old experiment
            return ConditionCompat(cond)

    def get_num_frames(self, seconds):
        """returns the number of frames that should be recorded for the given seconds"""
        return seconds / self._dt

    def get_experiment_metadata(self):
        """a list of dictionaries, each dict containing the metadata for the experiment"""
        return self._metadata

    def get_spanned_results(self):
        """
        returns a dict of object ids whose trajectories span multiple conditions.
        the values in the dict are a list of 2-tuples
            (condition,nframes)
        """
        return {}

    def enable_debug(self):
        self._enable_debug = True

    def disable_debug(self):
        self._enable_debug = False

    def enable_warn(self):
        self._enable_warn = True

    def disable_warn(self):
        self._enable_warn = False

    def get_results(self):
        """Returns all data for all conditions that passed the configured filters

        Returns:
            A two-tuple.

            the first retured variable is a nested dict of the following 
            structure

            results = {
               condition:{
                   df:[dataframe,...],
                   start_obj_ids:[(x0,y0,obj_id,framenumber0,time0),...],
                   count:n_frames,
                   uuids:[uuid,...],
               }

            The second is dt

        """
        r = collections.OrderedDict()
        for c in sorted(self._results, key=self.get_condition_name):
            r[c] = self._results[c]
        return r, self.dt

    def get_one_result(self, obj_id, condition=None, framenumber0=None):
        """
        Get the data associated with a single object_id

        returns: (dataframe, dt, (x0,y0,obj_id,framenumber0,time0))
        """
        spanned = self.get_spanned_results()
        if obj_id in spanned:
            if (condition is None) and (framenumber0 is None):
                raise ValueError("obj_id %s exists in multiple conditions: %s" % (obj_id,','.join(self.get_condition_name(d[0]) for d in spanned[obj_id])))

        for i,(current_condition,r) in enumerate(self._results.iteritems()):
            for df,(x0,y0,_obj_id,_framenumber0,time0),uuid in zip(r['df'], r['start_obj_ids'],r['uuids']):
                if _obj_id == obj_id:
                    if (condition is None) and (framenumber0 is None):
                        return df,self._dt,(x0,y0,_obj_id,_framenumber0,time0,current_condition,uuid)
                    elif (condition is not None) and (current_condition == condition):
                        return df,self._dt,(x0,y0,_obj_id,_framenumber0,time0,current_condition,uuid)
                    elif (framenumber0 is not None) and (_framenumber0 == framenumber0):
                        return df,self._dt,(x0,y0,_obj_id,_framenumber0,time0,current_condition,uuid)

        raise ValueError("No such obj_id: %s (condition: %s framenumber0: %s)" % (obj_id, condition, framenumber0))

    def get_results_dataframe(self, cols, fill=True, extra_col_cb=None):
        """
        Return a single dataframe containing the supplied columns from 
        all trials.

        This is a long-form dataframe with the additional colums
            ['condition','condition_name','obj_id','uuid','framenumber0']

        To extract trials one can do something like
            df.groupby(['condition','uuid','obj_id','framenumber0'])

        For aggregate meansures of those trials one can do
            df = df.groupby(['condition','uuid','obj_id','framenumber0'])['velocity'].mean().reset_index()
            sns.boxplot(x="condition",y="velocity",data=df)

        Parameters
        ----------
        cols: list of columns from the individual dataframes
        to put in the merged dataframe. If cols is None, return a dataframe with
        all availalable columns

        fill: forward and backward fill individual dataframes

        extra_col_cb: a function to call each dataframe that may add
        extra columns. signature is

            def add_ts_things(combine, data, uuid, current_condition, oid, start_framenumber, df)

        """

        DEFAULT = ['condition','condition_name','obj_id','uuid','framenumber0', 'start_time']

        if cols is None:
            cols = set(self.get_result_columns()) - set(DEFAULT)
        all_cols = set(itertools.chain.from_iterable((DEFAULT,cols)))

        data = {k:[] for k in all_cols}

        for current_condition,r in self._results.iteritems():
            cond_name = self.get_condition_name(current_condition)

            if not r['count']:
                continue

            for df,uuid,(start_x, start_y, oid, start_framenumber, start_time) in zip(r['df'], r['uuids'],r['start_obj_ids']):
                if fill:
                    _df = df.ffill().bfill()
                else:
                    _df = df

                for col in cols:
                    data[col].extend(_df[col].values)

                n = len(_df)
                data['condition'].extend(itertools.repeat(current_condition, n))
                data['condition_name'].extend(itertools.repeat(cond_name, n))
                data['obj_id'].extend(itertools.repeat(oid, n))
                data['uuid'].extend(itertools.repeat(uuid, n))
                data['framenumber0'].extend(itertools.repeat(start_framenumber, n))
                data['start_time'].extend(itertools.repeat(start_time, n))

                if extra_col_cb is not None:
                    extra_col_cb(self, data, uuid, current_condition, oid, start_framenumber, _df)

        return pd.DataFrame(data)

    def get_trials_dataframe(self,
                             add_dt=False,
                             startf_name='frame0',
                             endf_name='endf',
                             startt_name='time0'):
        """
        Aggregate the results to a trial-level (one trial per row) tidy dataframe.
        """
        # See also flydata conversions

        results, dt = self.get_results()

        dfs_stuff = []
        for cond, cond_dict in results.items():
            dfs = cond_dict['df']
            sois = cond_dict['start_obj_ids']
            uuids = cond_dict['uuids']
            for uuid, (x0, y0, obj_id, framenumber0, time0), df in zip(uuids, sois, dfs):
                dfs_stuff.append((uuid, obj_id, framenumber0, time0, len(df), df))
        df = pd.DataFrame(dfs_stuff, columns=['uuid', 'oid',
                                              startf_name, startt_name, 'length_f', 'series'])
        if endf_name is not None:
            df[endf_name] = df[startf_name] + df['length_f']  # half-open interval
        if add_dt:
            df['dt'] = dt  # no support for multiple dt in combine ATM

        return df

    def get_obj_ids_sorted_by_length(self):
        """
        Get a sorted list of the longest trajectories
        returns: a dictionary condition:[(obj_id,len),...]
        """
        best = {}
        for i,(current_condition,r) in enumerate(self._results.iteritems()):
            if not r['count']:
                continue
            for df,(x0,y0,obj_id,framenumber0,time0) in zip(r['df'], r['start_obj_ids']):
                try:
                    best[current_condition][obj_id] = len(df)
                except KeyError:
                    best[current_condition] = {obj_id:len(df)}

        sorted_best = {c:sorted(best[c].items(), key=operator.itemgetter(1), reverse=True) for c in self.get_conditions()}
        return sorted_best

    def get_result_columns(self):
        """get the names of the columns in the combined dataframe"""
        for current_condition,r in self._results.iteritems():
            for df in r['df']:
                if not r['count']:
                    continue
                return list(df.columns)
        return []

    def filter_trials(self, filter_func=lambda trial: True):
        """Returns a new combine object, filtering out trials that do not meet the filtering condition.
        Makes a best-effort to keep everything else the same.

        Parameters
        ----------
        filter_func: function (condition, df, start_obj_id, uuid, dt) -> boolean, default _ -> True (keep all)
          The predicate to evaluate on each trial data; only trials that evaluate to True are kept

        Returns
        -------
        A shallow-copy of the combine object, but keeping only the trials
        (be careful if changing any other mutable member, as changes would propagate to the original Combine object)
        """
        # Copy
        filtered = copy.copy(self)
        filtered._results = {}
        # Filter
        results = self._results  # warning, do not use get_results, it renames conds
        for cond_name, cond_trials in results.iteritems():
            dfs = []
            start_obj_ids = []
            uuids = []
            for df, soid, uuid in izip(
                    cond_trials['df'], cond_trials['start_obj_ids'], cond_trials['uuids']):
                if filter_func(cond_name, df, soid, uuid, self._dt):
                    dfs.append(df)
                    start_obj_ids.append(soid)
                    uuids.append(uuid)

            if dfs:
                filtered._results[cond_name] = {
                    'df': dfs,
                    'start_obj_ids': start_obj_ids,
                    'count': len(dfs),
                    'uuids': uuids
                }

        return filtered

    def close(self):
        pass

class _CombineFakeInfinity(_Combine):

    CONDITION_FMT_STRING = "tex%d/svg/1.0/1.0/adv/..."

    def __init__(self, **kwargs):
        _Combine.__init__(self, **kwargs)
        self._nconditions = kwargs.get('nconditions',1)
        self._ntrials = kwargs.get('ntrials', 100)
        self._ninfinity = kwargs.get('ninfinity', 5)

        self._results = {}
        self._dt = 1/100.0
        self._t0 = time.time()
        self._tzname = 'UTC'

    def _add(self):
        obj_id = 1
        framenumber = 1

        for c in range(self._nconditions):
            cond = self.CONDITION_FMT_STRING % c

            try:
                self._results[cond]
            except KeyError:
                self._results[cond] = {"df":[],"start_obj_ids":[],"count":0, 'uuids':[]}
                self._skipped[cond] = 0
                self._condition_names[cond] = "tex%d" % c

            for t in range(self._ntrials):
                if obj_id == 1:
                    #make sure the first infinity is full and perfect
                    df = self.get_fake_infinity(self._ninfinity,0,0,0,framenumber,0,0,self._dt)
                else:
                    df = self.get_fake_infinity(
                                n_infinity=self._ninfinity,
                                random_stddev=0.008,
                                x_offset=0.1 * (random.random() - 0.5),
                                y_offset=0.1 * (random.random() - 0.5),
                                frame0=framenumber,
                                nan_pct=1,
                                latency=10,
                                dt=self._dt
                    )
                    #take some random last number of trajectory
                    df = df.tail(random.randrange(10,len(df)))

                if len(df) < self.min_num_frames:
                    self._skipped[cond] += 1
                    continue

                dt = self._dt
                self.features.process(df, dt)

                first = df.irow(0)
                last = df.irow(-1)
                f0 = first.name

                self._results[cond]["df"].append(df)
                self._results[cond]["count"] += 1
                self._results[cond]["start_obj_ids"].append(
                        (first['x'],first['y'],obj_id,f0,self._t0 + (f0 * self._dt))
                )
                self._results[cond]['uuids'].append(None)

                obj_id += 1
                framenumber += len(df)

    @staticmethod
    def get_fake_infinity_trajectory(n_infinity, random_stddev, x_offset, y_offset, frame0, npts=200):
        def get_noise(d):
            if random_stddev:
                return (np.random.random(len(d)) - 0.5) * random_stddev
            else:
                return np.zeros_like(d)

        pi = np.pi
        leaft = np.linspace(-pi/4.,pi/4., npts//2)
        leaftrev = (leaft-pi)[::-1] #reverse

        theta = np.concatenate( (leaft, leaftrev[1:]) )

        r = np.cos(2*theta)

        x = np.concatenate( ([0], r*np.cos( theta )) )
        y = np.concatenate( ([0], r*np.sin( theta )) )

        x *= 0.4
        y *= 0.8

        x += x_offset
        y += y_offset

        N = len(x)

        z = np.zeros_like(x)
        z[:] = 0.7

        ratio = np.array(range(0,N)) / float(N)

        if n_infinity > 1:
            df = pd.DataFrame({
                    "x":np.concatenate( [x + get_noise(x) for i in range(n_infinity) ] ),
                    "y":np.concatenate( [y + get_noise(y) for i in range(n_infinity) ] ),
                    "z":np.concatenate( [z + get_noise(z) for i in range(n_infinity) ] ),
                    "ratio":np.concatenate( [ratio + get_noise(ratio) for i in range(n_infinity) ] )},
                    index=range(frame0,frame0+(N*n_infinity))
            )
        else:
            df = pd.DataFrame({
                    "x":x + get_noise(x),
                    "y":y + get_noise(y),
                    "z":z + get_noise(z),
                    "ratio":ratio + get_noise(ratio)},
                    index=range(frame0,frame0+N)
            )

        return df

    @staticmethod
    def get_fake_infinity(n_infinity, random_stddev, x_offset, y_offset, frame0, nan_pct, latency, dt, npts=200):
        df = _CombineFakeInfinity.get_fake_infinity_trajectory(n_infinity, random_stddev, x_offset, y_offset, frame0, npts)

        #despite these being recomputed later, we need to get them first
        #to make sure rrate is correlated to dtheta, we remove the added colums later
        m = afeat.MultiFeatureComputer('dtheta')
        m.process(df, dt)

        #add some uncorrelated noise to rrate
        rrate = (df['dtheta'].values * 10.0) + (0.0 * (np.random.random(len(df)) - 0.5))

        if latency > 0:
            rrate = np.concatenate( (rrate[latency:],np.random.random(latency)) )

        #rotation rate comes from a csv file, with a lower freq, so trim some % of the data
        #to simulate missing values
        trim = np.random.random(len(rrate)) < (nan_pct / 100.0)
        rrate[trim] = np.nan

        df['rotation_rate'] = rrate
        df['v_offset_rate'] = np.zeros_like(rrate)

        #remove the added colums
        for c in m.get_columns_added():
            del df[c]

        return df

    def add_from_args(self, args):
        self._args_to_configuration(args)

        self._lenfilt = args.lenfilt
        self._maybe_add_tfilt(args)
        self.plotdir = args.outdir if args.outdir else os.getcwd()
        self.csv_file = "test"

        self._add()

    def add_test_infinities(self):
        self._lenfilt = 0
        self.plotdir = "/tmp"
        self.csv_file = "test"
        self._add()

class CombineCSV(_Combine):

    csv_file = ''

    def __init__(self, **kwargs):
        _Combine.__init__(self, **kwargs)
        #when we add multiple csv files we have to ensure the frame_number is
        #monotonically increasing. we are lucky that the convention for csv
        #files generated by the tethered rig is that lock_obj = int(time.time())
        self._last_framenumber = 0
        self._df = None
        self._csv_suffix = kwargs.get("csv_suffix")


    def add_from_args(self, args, csv_suffix=None):
        self._args_to_configuration(args)

        if not csv_suffix:
            csv_suffix = self._csv_suffix

        self._idfilt = args.idfilt
        self._maybe_add_tfilt(args)

        if args.uuid:
            for uuid in args.uuid:
                fm = autodata.files.FileModel(basedir=args.basedir)
                fm.select_uuid(uuid)
                try:
                    csv_file = fm.get_file_model(csv_suffix).fullpath
                except autodata.files.NoFile:
                    if len(args.uuid) == 1:
                        raise
                    else:
                        continue
                self.add_csv_file(csv_file, args.lenfilt)
                self._update_plot_dir(args,uuid)
        else:
            uuid = self.add_csv_file(args.csv_file, args.lenfilt)
            self._update_plot_dir(args,uuid)

    def add_csv_file(self, csv_file, lenfilt=None):
        self._debug("IO:     reading %s" % csv_file)
        self.csv_file = csv_file

        df = pd.DataFrame.from_csv(self.csv_file,index_col="framenumber")
        assert 'lock_object' in df.columns

        # uuids from CSV
        uuid = None
        if 'exp_uuid' in df:
            if df['exp_uuid'].nunique() == 1:
                uuid = df['exp_uuid'].dropna().unique()[0]
        if uuid is None:
            # FIXME: some csvs lack the exp_uuid for some initial observations:
            #        e.g. cdb7a1ac94f711e4bb6cbcee7bdac270
            #        diagnose why (race condition?) and write defensive uuid readers...
            self._warn('cannot infer a unique UUID for %s' % self.csv_file)

        self._debug("IO:     csv uuid %s" % uuid)

        df = df.fillna(method="pad")
        df['time'] = df['t_sec'] + (df['t_nsec'] * 1e-9)

        dt = (df['time'].values[-1] - df['time'].values[0]) / len(df)

        if self._dt is None:
            self._dt = dt
            self._lenfilt = lenfilt
        else:
            #check the new csv file was recorded with the same timebase
            assert abs(dt-self._dt) < 1e-4

        for obj_id, lodf in df.groupby('lock_object'):

            # Continuous grouping, see CombineH5WithCSV
            for _, odf in lodf.groupby((lodf['condition'] != lodf['condition'].shift()).cumsum()):

                #start of file
                if odf['condition'].count() == 0:
                    continue

                if obj_id in (IMPOSSIBLE_OBJ_ID,IMPOSSIBLE_OBJ_ID_ZERO_POSE):
                    continue

                if self._idfilt and (obj_id not in self._idfilt):
                    continue

                if not self._df_ok(odf):
                    continue

                assert odf['condition'].nunique() == 1, 'A single trial must not span more than one condition'

                cond = odf['condition'].iloc[0]
                cond = analysislib.fixes.normalize_condition_string(cond)

                if cond not in self._results:
                    self._results[cond] = {'df':[],'start_obj_ids':[],'count':0, 'uuids':[]}
                    try:
                        self._condition_names[cond] = odf['condition_name'].iloc[0]
                    except:
                        pass

                dt = self._dt
                self.features.process(df, dt)

                self._results[cond]['df'].append(odf)
                self._results[cond]['start_obj_ids'].append(self._get_result(odf))
                self._results[cond]['count'] += 1

                self._results[cond]['uuids'].append(uuid)

        if self._df is None:
            self._df = df
        else:
            last = df.last_valid_index() + self._last_framenumber + 1
            #increase the indexes of this df to be past the end of the last one
            df = df.set_index(df.index.values + last)
            #append / concat this at the end of the current dataframe
            self._df = pd.concat((self._df, df))
            self._last_framenumber = last

    def _df_ok(self, df):
        if len(df) < self.min_num_frames:
            return False
        else:
            return True

    def _get_result(self, df):
        ser = df.irow(0)
        return ser['x'],ser['y'],ser['lock_object'],ser.name,ser['time'],'',''

    def get_one_result(self, obj_id, condition=None, framenumber0=None):
        df = self._df[self._df['lock_object'] == obj_id]
        if not len(df):
            raise ValueError("object %s not found" % obj_id)

        if not self._df_ok(df):
            raise ValueError("result is too short")

        return df,self._dt,self._get_result(df)

class CombineH5(_Combine):

    h5_file = ''

    def __init__(self, **kwargs):
        _Combine.__init__(self, **kwargs)
        self._dt = None
        self._h5 = None

    def add_from_args(self, args):
        self._args_to_configuration(args)

        self._maybe_add_tfilt(args)

        if args.uuid:
            uuid = args.uuid[0]
            fm = autodata.files.FileModel(basedir=args.basedir)
            fm.select_uuid(uuid)
            h5_file = fm.get_file_model("simple_flydra.h5").fullpath
        else:
            h5_file = args.h5_file

        self.add_h5_file(h5_file)

    def add_from_uuid(self, uuid, *args, **kwargs):
        fm = autodata.files.FileModel()
        fm.select_uuid(uuid)
        h5_file = fm.get_file_model("simple_flydra.h5").fullpath
        self.add_h5_file(h5_file)

    def add_h5_file(self, h5_file):
        self._debug("IO:     reading %s" % h5_file)

        warnings = {}

        self.h5_file = h5_file

        self._h5 = tables.openFile(h5_file, mode='r')

        self._trajectories = self._get_trajectories(self._h5)
        dt = 1.0/self._trajectories.attrs['frames_per_second']

        self._trajectory_start_times = self._h5.root.trajectory_start_times
        tzname = self._h5.root.trajectory_start_times.attrs['timezone']

        if self._dt is None:
            self._dt = dt
            self._tzname = tzname
        else:
            assert dt == self._dt
            assert tzname == self._tzname

    def get_one_result(self, obj_id, condition=None, framenumber0=None):
        query = "obj_id == %d" % obj_id

        traj = self._trajectories.readWhere(query)
        start = self._trajectory_start_times.readWhere(query)

        _,tsec,tnsec = start[0]
        t0 = tsec+(tnsec*1e-9)

        df = pd.DataFrame(
                {i:traj[i] for i in 'xyz'},
                index=traj['framenumber']
        )

        dt = self._dt
        self.features.process(df, dt)

        return df,self._dt,(traj['x'][0],traj['y'][0],obj_id,traj['framenumber'][0],t0,'','')

    def close(self):
        if self._h5 is not None:
            self._h5.close()

class CombineH5WithCSV(_Combine):
    """
    Combines the data from a smoothed (simple_flydra) h5 file with
    a csv file based on the framenumber
    """

    csv_file = ''
    h5_file = ''

    def __init__(self, *csv_cols, **kwargs):
        _Combine.__init__(self, **kwargs)
        self._csv_suffix = kwargs.get("csv_suffix")

        #use this for keeping track of results that span multiple conditions
        self._results_by_condition = {}

        # split bookkeeping
        self._split_bookeeping = {}    # {(uuid, oid, startf) -> (split_num, reason_for_split)}

    def split_reason(self, uuid, oid, startf):
        """Returns a two tuple for the splitting reason (why we changed trail) for the trial at hand.
          - The first element is the split-number
          - The second element is one of:
            - 'marker': the controller explicitly says "new trial"
            - 'condition': condition switch
            - 'oid': oid switch
            - 'frame-diff': the next observation was too far (magic number in the code... ;-))
            - 'last': the trial was the last one in the CSV

        Raises exception if the trial cannot be found.
        """
        try:
            return self._split_bookeeping[(uuid, oid, startf)]
        except KeyError:
            raise Exception('The trial (%s, %d, %d) is not in the books' % (uuid, oid, startf))

    def get_split_reason_dataframe(self):
        d = {k:[] for k in ('uuid', 'oid', 'startf', 'split_num', 'reason_for_split')}
        for (uuid, oid, startf), (split_num, reason_for_split) in self._split_bookeeping.iteritems():
            d['uuid'].append(uuid)
            d['oid'].append(oid)
            d['startf'].append(startf)
            d['split_num'].append(split_num)
            d['reason_for_split'].append(reason_for_split)
        return pd.DataFrame(d)

    def add_from_uuid(self, uuid, csv_suffix=None, **kwargs):
        """Add a csv and h5 file collected from the experiment with the
        given uuid.

        Returns an argparse Namespace containing the configuration arguments.
        """
        if 'args' in kwargs:
            args = kwargs['args']
            args.uuid = [uuid]
        else:
            kwargs['uuid'] = uuid  # note: this is side effect free, as python warrants kwargs is a fresh copy
            parser,args = analysislib.args.get_default_args(**kwargs)

        self.add_from_args(args, csv_suffix=csv_suffix)

        return args

    def populate_from_dictionary(self, d):
        self._results = d['results']
        self._dt = d['dt']
        self._skipped = d['skipped']
        self.csv_file = d['csv_file']   # for plot names
        self._conditions = d['conditions']
        self._condition_names = d['condition_names']
        self._metadata = d['metadata']

    def add_from_args(self, args, csv_suffix=None):
        """Add possibly multiple csv and h5 files based on the command line
        arguments given
        """
        arena = analysislib.args.get_arena_from_args(args)
        for c in arena.get_filter_columns():
            self.add_feature(column_name=c)

        self._args_to_configuration(args)

        cache_error = False
        if args.cached:

            try:
                cached_dict = self._read_cache_file()
            except CacheError:
                cached_dict = None
                cache_error = True
                
            if cached_dict is not None:
                self.populate_from_dictionary(cached_dict)
                uuids = set(itertools.chain.from_iterable(self._results[cond]['uuids']
                                                          for cond in self._results))
                self._update_plot_dir(args, None, uuids)
                return

        if not csv_suffix:
            csv_suffix = self._csv_suffix

        self._maybe_add_tfilt(args)

        # fail right at the start if we won't be able to save the cache
        if not set_permissions():
            self._warn('Cannot set strawlab-friendly isilon permissions (gid=2046, umask=002)')
            if not args.get('ignore_permission_errors', False):
                raise CacheError('wrong umask or gid here, fix or use --ignore-permission-errors')

        if args.uuid:
            for uuid in args.uuid:
                fm = autodata.files.FileModel(basedir=args.basedir)
                fm.select_uuid(uuid)
                try:
                    csv_file = fm.get_file_model(csv_suffix).fullpath
                    h5_file = fm.get_file_model("simple_flydra.h5").fullpath
                except autodata.files.NoFile:
                    if len(args.uuid) == 1:
                        raise
                    else:
                        continue
                self.add_csv_and_h5_file(csv_file, h5_file, args)
                self._update_plot_dir(args,uuid)
        else:
            csv_file = args.csv_file
            h5_file = args.h5_file
            uuid = self.add_csv_and_h5_file(csv_file, h5_file, args)
            self._update_plot_dir(args,uuid)

        if cache_error or (not os.path.isfile(self.get_cache_pkl_path())):
            if args.cached:
                self.save_cache()
        elif args.recache:
            self.save_cache()

    def get_spanned_results(self):
        spanned = {}
        for oid,details in self._results_by_condition.iteritems():
            if len(details) > 1:
                spanned[oid] = details
        return spanned

    def infer_uuid(self, args_uuids, csv_df, csv_fname, h5_fname=None):
        """Given args and the csv_df, try to infer a unique UUID."""
        uuid = None
        uuids_from_csv = uuids_from_experiment_csv(csv_df)  # can be more than one
        self._debug("IO:     csv uuid(s) %s" % ','.join(uuids_from_csv))

        if h5_fname is not None:
            uuids_from_flydra = uuids_from_flydra_h5(h5_fname, logger=self._warn)  # can be more than one
            self._debug("IO:     h5 uuid(s) %s" % ','.join(uuids_from_flydra))
            uuid_candidates_from_files = set(uuids_from_flydra) & set(uuids_from_csv)  # can be more than one
        else:
            uuid_candidates = set(uuids_from_csv)

        uuids_from_args = args_uuids if args_uuids is not None else []  # can be more than one

        if len(uuids_from_args) == 1:
            uuid_candidate = uuids_from_args[0]
            if uuid_candidate not in uuid_candidates_from_files:  # we do not want to recover from this
                raise Exception('uuid %s not present in the csv %s' % (uuid_candidate, csv_fname))
            uuid = uuid_candidate
        elif len(uuids_from_args) > 1:
            uuid_candidates = uuid_candidates_from_files & set(uuids_from_args)
            if len(uuid_candidates) != 1:
                self._warn('Chosing none of possible uuids: %s' % ' '.join(sorted(uuid_candidates)))
            else:
                uuid = uuid_candidates.pop()
        else:
            uuid_candidates = uuid_candidates_from_files
            if len(uuid_candidates) != 1:
                self._warn('Chosing none of possible uuids: %s' % ' '.join(sorted(uuid_candidates)))
            else:
                uuid = uuid_candidates.pop()

        return uuid

    def _read_csv(self, csv_fname):
        """
        Reads the csv file as a dataframe.
        If memory ever is a problem, look here (and just read incrementally).
        """

        # N.B. we only need self for emitting log messages

        self._debug("IO:     reading %s" % csv_fname)

        try:
            csv = pd.read_csv(csv_fname, na_values=('None',),
                              error_bad_lines=False,
                              dtype={'framenumber': int,
                                     'condition': str,
                                     'exp_uuid': str,
                                     'flydra_data_file': str})
        except:  # Fixme: narrow down except
            self._warn("ERROR: possibly corrupt csv. Re-parsing %s" % csv_fname)
            # protect against rubbish in the framenumber column
            csv = pd.read_csv(csv_fname, na_values=('None',),
                              error_bad_lines=False,
                              low_memory=False,
                              dtype={'framenumber': float,
                                     'condition': str,
                                     'exp_uuid': str,
                                     'flydra_data_file': str})
            csv = csv.dropna(subset=['framenumber'])
            csv['framenumber'] = csv['framenumber'].astype(int)
        return csv

    def read_conditions(self, csv_fname):
        this_exp_conditions = {}
        path, fname = os.path.split(csv_fname)
        try:
            fn = os.path.join(path, fname.split('.')[0] + '.condition.yaml')
            with open(fn) as f:
                self._debug("IO:     reading %s" % fn)
                this_exp_conditions = yaml.safe_load(f)
                try:
                    del this_exp_conditions['uuid']
                except KeyError:
                    pass
        except:  # FIXME
            pass
        return this_exp_conditions

    def read_metadata(self, uuid, csv_fname):
        this_exp_conditions = self.read_conditions(csv_fname)
        this_exp_metadata = {}
        path, fname = os.path.split(csv_fname)
        try:
            # get it from the database, if it fails, try from the yaml
            fn = os.path.join(path, fname.split('.')[0] + '.experiment.yaml')
            try:
                self._debug("IO:     reading from database")
                _, arena, this_exp_metadata = find_experiment(uuid)
                this_exp_metadata['arena'] = arena
                self._metadata.append(this_exp_metadata)  # FIXME: we should restrict combine to just one exp at a time
                # try to update the yaml
                # (we need to tell to people these yaml are read-only, subject to change for them)
                try:
                    try:
                        os.unlink(fn)
                    except:
                        pass
                    with open(fn, 'w') as f:
                        yaml.safe_dump(this_exp_metadata, f, default_flow_style=False)
                        self._debug("IO:     wrote %s" % fn)
                except Exception as e:
                    self._debug("IO:     ERROR writing %s\n%s" % (fn, e))
            except Exception as e:
                self._debug("IO:     ERROR reading from database\n%s" % e)
                with open(fn) as f:
                    self._debug("IO:     reading %s" % fn)
                    self._metadata.append(yaml.safe_load(f))
        except Exception as e:
            self._debug("IO:     ERROR reading metadata\n%s" % e)

        if this_exp_metadata is None:
            this_exp_metadata = {}

        this_exp_metadata['csv_file'] = self.csv_file  # FIXME: this will get us into troubles, not needed
        this_exp_metadata['h5_file'] = self.h5_file  # FIXME: this will get us into troubles, not needed
        this_exp_metadata['conditions'] = this_exp_conditions

        return this_exp_conditions, this_exp_metadata

    def _read_h5(self, h5_file, reindex):
        # open h5 file
        self._debug("IO:     reading %s" % h5_file)
        h5 = tables.openFile(h5_file, mode='r+' if reindex else 'r')
        trajectories = self._get_trajectories(h5)
        dt = 1.0 / trajectories.attrs['frames_per_second']
        tzname = h5.root.trajectory_start_times.attrs['timezone']

        # workaround old pytables not able to read unicode
        try:
            pytz.timezone(tzname)
        except UnicodeDecodeError:
            self._warn("ERROR: PYTABLES UNICODE DECODE ERROR. UPGRADE PYTABLES")
            tzname = 'CET'

        if self._dt is None:
            self._dt = dt
            self._tzname = tzname
        else:
            assert dt == self._dt
            assert tzname == self._tzname

        return h5, trajectories

    def add_csv_and_h5_file(self, csv_fname, h5_file, args):
        """Add a single csv and h5 file"""

        # Update self.csv_file for every csv file, even if we contain
        # data from many. This for historical reasons as the csv file is used
        # as the basename for generated plots, so saving a few test
        # analyses with --outdir /tmp/ gives distinct named plots
        self.csv_file = csv_fname
        self.h5_file = h5_file

        # Read full csv into memory
        csv = self._read_csv(csv_fname)

        # infer uuid
        uuid = self.infer_uuid(args.uuid, csv, csv_fname, h5_file)

        # read conditions and experimental metadata
        this_exp_conditions, this_exp_metadata = self.read_metadata(uuid, csv_fname)

        # infer arena
        arena = analysislib.args.get_arena_from_args(args)

        # find single applicable data fixer
        fix = analysislib.fixes.load_csv_fixups(**this_exp_metadata)
        if fix.active:
            self._debug("FIX:    fixing data %s" % fix)

        # open simple-flydra h5 file (for sanity checks and dt/tz population)
        h5, trajectories = self._read_h5(h5_file, args.reindex)

        # setup lenfilt
        if self._lenfilt is None:
            self._lenfilt = args.lenfilt
        else:
            assert self._lenfilt == args.lenfilt  # FIXME: this might not be correct under certain lifecycles

        # frames start offset
        frames_start_offset = int(args.trajectory_start_offset / self._dt)

        # minimum length of 2 to prevent later errors calculating derivitives
        dur_samples = max(2, self.min_num_frames)

        # aliases
        results = self._results
        skipped = self._skipped

        #
        # split trials
        #
        # The CSV (output from the controller) indicates how to segment trials, although not
        # always in an unambiguous way. We should force stimuli writers to do the right thing,
        # letting them know what that right is.
        #
        # At the moment, a trial end is determined by either:
        #   - a lock_object (oid) change
        #   - or a condition change
        # A trial should have unique oid and condition
        #
        # Lines can be written in our CSVs in 3 different situations:
        #   - a regular observation during a experiment
        #   - a marker row for a condition change (present in all CSVs)
        #   - a marker row for a lock_object change or loss (present only in newer CSVs)
        #
        # Marker rows have either oid=IMPOSSIBLE_OBJ_ID or oid=IMPOSSIBLE_OBJ_ID_ZERO_POSE
        # The beginning of the file might contain garbage (because we do not sync with flydra or exp. start).
        # Framenumber is *not* warrantee to be monotonically increasing (although it should be close)
        #
        # Newer CSVs are better designed because using these marker observations is
        # *the only correct way to segment trials*.
        #
        # For old CSVs we need a heuristic to account for the hopefully rare case
        # (less frequent in the more animals are in the arena) in which
        # the same lock_object would be given two trials within the same condition
        # realisation (meaning within the time between two consecutive switches
        # to different conditions) and with no other oid given a trial in the middle.
        # Such heuristic, based framenumbers, should allow us to split these cases
        # (instead of time-travelling).
        #

        def iterative_groups(oid, condition, framenumber,
                             # these three keep track of last value, while not poluting outer namespace
                             # only codestyle warning in this whole function ATM, keep it like that! ;-)
                             trial_count=[0],
                             last_oid=[csv.iloc[0]['lock_object']],
                             last_condition=[csv.iloc[0]['condition']],
                             last_framenumber=[csv.iloc[0]['framenumber']],
                             framenumber0=[csv.iloc[0]['framenumber']],
                             min_frames_diff_split=10):
            # for bookeeping
            if framenumber0[0] is None:
                framenumber0[0] = last_framenumber[0]
            # new style, marker rows
            if oid == IMPOSSIBLE_OBJ_ID or oid == IMPOSSIBLE_OBJ_ID_ZERO_POSE:
                trial_count[0] += 1
                self._split_bookeeping[(uuid, last_oid[0], framenumber0[0])] = (trial_count[0], 'marker')
                framenumber0[0] = None
                return -1
            # old style, change of oid (this would never happen on newer CSV versions)
            if oid != last_oid[0]:
                self._split_bookeeping[(uuid, last_oid[0], framenumber0[0])] = (trial_count[0], 'oid')
                framenumber0[0] = None
                trial_count[0] += 1
            # old style, change of condition (this would never happen on newer CSV versions)
            if condition != last_condition[0]:
                self._split_bookeeping[(uuid, last_oid[0], framenumber0[0])] = (trial_count[0], 'condition')
                framenumber0[0] = None
                trial_count[0] += 1
            # heuristic for old CSVs (this would never happen on newer CSV versions)
            if framenumber - last_framenumber[0] > min_frames_diff_split:
                self._split_bookeeping[(uuid, last_oid[0], framenumber0[0])] = (trial_count[0], 'frame-diff')
                framenumber0[0] = None
                trial_count[0] += 1
            last_oid[0] = oid
            last_condition[0] = condition
            last_framenumber[0] = framenumber
            return trial_count[0]

        # timeline = csv.apply(iterative_groups, axis=1)  # apply over rows is real slow
        timeline = np.array(
            [iterative_groups(oid, cond, framenumber) for oid, cond, framenumber in
             izip(csv['lock_object'], csv['condition'], csv['framenumber'])])
        # compress intervals, save... useful?

        # complete the bookkeeping for the last trial
        # do not use "if missing, then last" because that would hide bad calls to "split_reason"
        last_trial_index = timeline.max()
        last_trial = csv[timeline == last_trial_index].iloc[0]
        self._split_bookeeping[(uuid, last_trial['lock_object'], last_trial['framenumber'])] = \
            (last_trial_index, 'last')

        # find condition switches, save
        self._condition_switches[uuid] = condition_switches_from_controller_csv(csv)

        for trial_num, csv_df in csv.groupby(timeline):

            # start of file?
            if csv_df['condition'].count() == 0:
                continue

            if not csv_df['lock_object'].nunique() == 1:
                raise Exception('CSV problem, more than one object id in the same trial:\n\ttrial=%d oids=(%s) %s' %
                                (trial_num, ','.join(map(str, csv_df['lock_object'].unique())), csv_fname))

            oid = csv_df['lock_object'].iloc[0]

            # controller marker observations group?
            if oid in (IMPOSSIBLE_OBJ_ID, IMPOSSIBLE_OBJ_ID_ZERO_POSE):
                continue

            # sometimes flydra crashes and I restart it while leaving the node running. in that
            # case the csv can contain references to two h5 files and two uuids. because
            # flydra re-uses object ids there can be confusion as to which trial an object id
            # refers. if combine is explictly constructed from a single uuid then only query the
            # h5 file if the csv says it should be present
            if uuid is not None:
                exp_uuids = csv_df['exp_uuid'].dropna().unique()
                if not len(exp_uuids):
                    # normal case, the first few rows before a uuid was assigned.
                    # assume everything is ok....
                    pass
                else:
                    # I can perform stricter checks because in the general case
                    # (i.e. not when someone has done add_uuid_to_csv) because
                    # nodelib writes the uuid on every line so we can check if we should
                    # query the h5 file
                    if len(exp_uuids) > 1:
                        self._warn("WARN: object id %d in multiple possible h5 files (%s)" % (oid, ','.join(exp_uuids)))
                        continue

                    # length of exp_uuids must be 1
                    if uuid not in exp_uuids:
                        self._warn("SKIP: object id %d in another h5 file (%s)" % (oid, exp_uuids[0]))
                        continue

            if not csv_df['condition'].nunique() == 1:
                raise Exception('CSV problem, more than one condition in the same trial:\n\ttrial=%d oids=(%s) %s' %
                                (trial_num, ','.join(csv_df['condition'].unique()), csv_fname))

            cond = csv_df['condition'].iloc[0]

            # do we want this object?
            if args.idfilt and (oid not in args.idfilt):
                continue

            original_condition = cond

            # fix and normalise condition strings
            if fix.active:
                cond = fix.fix_condition(cond)
            cond = analysislib.fixes.normalize_condition_string(cond)
            fixed_condition = cond

            condition_name = None
            if cond not in results:
                results[cond] = dict(count=0,
                                     start_obj_ids=[],
                                     df=[],
                                     uuids=[])
                skipped[cond] = 0
                try:
                    condition_name = csv_df['condition_name'].iloc[0]
                except:
                    # old data with no condition name
                    pass

            original_condition_name = condition_name
            if condition_name:
                if fix.active:
                    condition_name = fix.fix_condition_name(condition_name)

            if (condition_name is not None) and (cond not in self._condition_names):
                self._condition_names[cond] = condition_name

            if (condition_name is not None) and (original_condition_name != condition_name):
                self._debug_once("FIX:    #%5d: condition name %s -> %s" %
                                 (oid, original_condition_name, condition_name))
                csv_df = csv_df.copy()  # use copy and not view
                csv_df['condition_name'].replace(original_condition_name, condition_name, inplace=True)

            # at this point all fixes that manipulate the condition string/name have
            # been applied, so we can go and save the condition configuration
            if cond not in self._conditions:
                self._conditions[cond] = this_exp_conditions.get(original_condition_name, {})

            r = results[cond]

            # the csv may be written at a faster rate than the framerate,
            # causing there to be multiple rows with the same framenumber.
            # find the last index for all unique framenumbers for this trial
            csv_df = csv_df.drop_duplicates(cols=('framenumber',), take_last=True)
            trial_framenumbers = csv_df['framenumber'].values

            # there is sometimes some erronous entries in the csv due to some race
            # conditions when locking onto new objects. While we guarentee that
            # framenumbers must be monotonically increasing, perform a weaker version
            # of that test now and check the last is greater than the first
            if (len(trial_framenumbers) > 2) and (trial_framenumbers[0] > trial_framenumbers[-1]):
                self._warn('WARN:   corrupt trial for obj_id %s' % oid)
                continue

            if original_condition != fixed_condition:
                self._debug_once("FIX:    #%5d: condition string %s -> %s" % (oid, original_condition, fixed_condition))
                csv_df = csv_df.copy()  # use copy and not view
                csv_df['condition'].replace(original_condition, fixed_condition, inplace=True)

            # get the comparable range of data from flydra
            if frames_start_offset != 0:
                start_frame = trial_framenumbers[0] + frames_start_offset
            else:
                start_frame = trial_framenumbers[0]

            # provided that the framenumber[-1] is greater than framenumber[0],
            # which was ensured in the test above, then if the 'start_frame'
            # is already past the end frame (due to the frames_start_offset)
            # then the trajectory is too short
            if start_frame > trial_framenumbers[-1]:
                self._debug('SKIP:   #%5d: 0 valid samples' % oid)
                continue

            # get trajectory data from flydra
            query = "(obj_id == %d) & (framenumber >= %d) & (framenumber <= %d)" % \
                    (oid, start_frame, trial_framenumbers[-1])

            if fix.active and (fix.identifier == 'OLD_CONFINEMENT_CSVS_ARE_BROKEN'):
                # sorry everyone
                query = "(obj_id == %d) & (framenumber >= %d)" % (oid, start_frame)
                self._warn_once("WARN: THIS DATA IS BROKEN IF OBJ_ID SPANS CONDITIONS")

            try:
                valid = trajectories.readWhere(query)
            except:
                self._warn("ERROR: PYTABLES CRASHED QUERY")
                continue
            validframenumber = valid['framenumber']

            n_samples = len(validframenumber)

            #
            # A combine data contract is that returned time series do not have holes.
            #
            # This makes combine skip trials for which data coming from flydra has
            # holes, which in turn would result in a time series dataframe with holes.
            # An example:
            #   uuid='15268c1c256a11e5abf760a44c2451e5', obj_id=501804,
            #        there is a large hole around frame 132149 in the simple.flydra.h5 file
            #
            # Note that this lets pass any data from flydra that misses frames only at
            # the beginning or at the end. These are handled later on.
            #
            framenumber_diff = validframenumber[1:] - validframenumber[:-1]
            if (framenumber_diff != 1).any():
                self._debug('SKIP:   flydra data has holes for obj_id %d start_frame %d' %
                            (oid, start_frame))
                self._skipped[cond] += 1
                continue

            # check if the query returned all the data it should have
            if n_samples < dur_samples:
                # but this happens quite often (due to estimation of positions) so
                # only print the error when the missing data is large enough for us to
                # skip the trial
                if (trial_framenumbers[-1] - start_frame) > dur_samples:
                    self._warn("WARN:   obj_id %d missing %d frames from h5 file\n        %s" %
                               (oid, trial_framenumbers[-1] - start_frame, query))
                # we could have made this check before the query, but sometimes
                # we get less data that we request from the h5 file
                self._debug('SKIP:   #%5d: %d valid samples' % (oid, n_samples))
                self._skipped[cond] += 1
                continue

            flydra_series = []
            for a in ('x', 'y', 'z', 'covariance_x', 'covariance_y', 'covariance_z'):
                try:
                    avalid = valid[a]
                    flydra_series.append(pd.Series(avalid, name=a, index=validframenumber))
                except ValueError:
                    self._warn_once('WARN: %s lacks %s data' % (h5_file, a))

            # we can now create a dataframe that has the flydra data, and the
            # original index of the csv dataframe
            framenumber_series = pd.Series(validframenumber, name='framenumber', index=validframenumber)
            flydra_series.append(framenumber_series)

            h5_df = pd.concat(flydra_series, axis=1)
            n_samples_before = len(h5_df)

            # compute those features we can (such as those that only need x,y,z)
            try:
                self.features.process(h5_df, self._dt)
            except Exception as e:
                self._skipped[cond] += 1
                self._warn("ERROR: could not compute features for oid %s (%s long)\n\t%s" % (oid, n_samples, e))
                continue

            # apply filters
            filter_cond, _ = arena.apply_filters(args, h5_df, self._dt)
            h5_df = h5_df.iloc[filter_cond]

            n_samples = len(h5_df)
            if n_samples < dur_samples:
                self._debug('FILT:   #%5d: %d/%d valid samples' % (oid, n_samples, len(valid)))
                self._skipped[cond] += 1
                continue
            if n_samples != n_samples_before:
                self._debug('TRIM:   #%5d: removed %d frames' % (oid, n_samples_before - n_samples))

            traj_start_frame = h5_df['framenumber'].values[0]
            traj_stop_frame = h5_df['framenumber'].values[-1]

            # another bug fixed here, we were using csv instead of fdf (hidden for ages as we almost never used this...)
            start_time = float(csv_df.head(1)['t_sec'] + (csv_df.head(1)['t_nsec'] * 1e-9))
            if not self._maybe_apply_tfilt_should_save(start_time):
                h5_df = None

            if h5_df is not None:

                n_samples = len(h5_df)
                span_details = (cond, n_samples)
                self._results_by_condition.setdefault(oid, []).append(span_details)

                self._debug('SAVE:   #%5d: %d samples (%d -> %d) (%s)' %
                            (oid,
                             n_samples,
                             traj_start_frame, traj_stop_frame,
                             self.get_condition_name(cond)))

                if self._index == 'framenumber':
                    # if the csv has been written at a faster rate than the
                    # flydra data then csv_df contains the last estimate in the
                    # csv for that framenumber (because drop_duplicates take_last=True)
                    # removes the extra rows and make a new framenumber index
                    # unique.
                    #
                    # an outer join allows the tracking data to have started
                    # before the csv (frames_start_offset)

                    # delete the framenumber from the h5 dataframe, it only
                    # duplicates what should be in the index anyway
                    del h5_df['framenumber']

                    # if there are any columns common in both dataframes the result
                    # seems to be that the concat resizes the contained values
                    # by adding an extra dimenstion.
                    # df['x'].values.ndim = 1 becomes = 2 (for version of
                    # pandas < 0.14). To work around this, remove any columns
                    # in the csv dataframe that exists in the h5 dataframe
                    common_columns = h5_df.columns & csv_df.columns
                    for c in common_columns:
                        self._warn_once('ERROR: renaming duplicated colum name "%s" to "_%s"' % (c, c))
                        cv = csv_df[c].values
                        del csv_df[c]
                        csv_df['_' + c] = cv

                    df = pd.concat((csv_df.set_index('framenumber'), h5_df),
                                   axis=1, join='outer')

                    # restore a framenumber column for API compatibility
                    df['framenumber'] = df.index.values

                    if df['x'].values.ndim > 1:
                        self._warn_once("ERROR: pandas merge added empty dimension to dataframe values")

                    # Because of the outer join, trim filter do not work
                    # (trimmed observations come back as haunting missing values)
                    # This is a quick workaround...
                    df = df.dropna(subset=['x'])
                    # TODO: check for holes

                elif (self._index == 'none') or (self._index.startswith('time')):

                    if self._index.startswith('time'):
                        # add a tns column
                        csv_df['tns'] = np.array((csv_df['t_sec'].values * 1e9) + csv_df['t_nsec'], dtype=np.uint64)

                    # we still must trim the csv dataframe by the trim conditions (framenumber)
                    csv_fns = csv_df['framenumber'].values
                    csv_fn0_idx = np.where(csv_fns >= traj_start_frame)[0][0]   # first frame
                    csv_fnN_idx = np.where(csv_fns <= traj_stop_frame)[0][-1]   # last frame

                    # in this case we want to keep all the rows (outer)
                    # but the two dataframes should remain sorted by
                    # framenumber because we use that for building a new time index
                    # if we resample
                    df = pd.merge(csv_df.iloc[csv_fn0_idx:csv_fnN_idx], h5_df,  # trim as filtered
                                  suffixes=("_csv", "_h5"),
                                  on='framenumber',
                                  left_index=False, right_index=False,
                                  how='outer', sort=True)

                    # in the time case we want to set a datetime index and optionally resample
                    if self._index.startswith('time'):
                        try:
                            _, resamplespec = self._index.split('+')
                        except ValueError:
                            resamplespec = None

                        if df['framenumber'][0] != traj_start_frame:
                            dfv = df['framenumber'].values
                            # now the df is sorted we can just remove the invalid data from the front
                            n_invalid_rows = np.where(dfv == traj_start_frame)[0][0]

                            self._warn("WARN: csv started %s rows before tracking (fn csv:%r... vs h5:%s, obj_id) %s"
                                       % (n_invalid_rows, dfv[0:3], traj_start_frame, oid))

                            df = df.iloc[n_invalid_rows:]

                        traj_start = h5.root.trajectory_start_times.readWhere("obj_id == %d" % oid)
                        tns0 = (traj_start['first_timestamp_secs'] * 1e9) + traj_start['first_timestamp_nsecs']
                        if tns0 == 0.0:
                            self._warn("WARN: trajectory start time of object_id %s is 0" % oid)
                        df['tns'] = ((df['framenumber'].values - traj_start_frame) * self._dt * 1e9) + tns0

                        df['datetime'] = df['tns'].values.astype('datetime64[ns]')
                        # any invalid (NaT) rows break resampling
                        df = df.dropna(subset=['datetime'])
                        df = df.set_index('datetime')
                        # TODO: check for holes

                        if resamplespec is not None:
                            df = df.resample(resamplespec, fill_method='pad')
                else:
                    raise Exception('Unknown index requested %s' % self._index)

                if fix.should_fix_rows:
                    for _ix, row in df.iterrows():
                        fixed = fix.fix_row(row)
                        for col in fix.should_fix_rows:
                            if col not in df.columns:
                                self._warn_once("ERROR: column '%s' missing from dataframe (are you resampling?)" % col)
                                continue
                            # modify in place
                            try:
                                df.loc[_ix, col] = fixed[col]
                            except IndexError as e:
                                self._warn("ERROR: could not apply fixup to obj_id %s (column '%s'): %s" %
                                           (oid, col, str(e)))
                if fix.should_fix_dataframe:
                    fix.fix_dataframe(df)

                stop_framenumber = df['framenumber'].dropna().values[-1]

                # the start time and the start framenumber are defined by the experiment,
                # so they come from the csv
                first = csv_df.irow(0)

                start_time = float(first['t_sec'] + (first['t_nsec'] * 1e-9))
                start_framenumber = int(first['framenumber'])
                # we could get this from the merged dataframe, but this is easier...
                # also, the >= is needed to make valid['x'][0] not crash
                # because for some reason sometimes we have a framenumber
                # in the csv (which means it must be tracked) but not the simple
                # flydra file....?
                #
                # maybe there is an off-by-one hiding elsewhere
                query = "(obj_id == %d) & (framenumber >= %d)" % (oid, start_framenumber)
                valid = trajectories.readWhere(query)
                start_x = valid['x'][0]
                start_y = valid['y'][0]

                # compute the remaining features (which might have come from the CSV)
                try:
                    dt = self._get_df_sample_interval(df) or self._dt

                    opts = {'uuid': uuid,
                            'obj_id': oid,
                            'start_framenumber': start_framenumber,
                            'stop_framenumber': stop_framenumber,
                            'start_time': start_time,
                            'stop_time': start_time + ((stop_framenumber - start_framenumber) * dt),
                            'condition': cond,
                            'condition_object': self.get_condition_object(cond)}

                    computed, not_computed, missing = self.features.process(df, dt, **opts)
                    if missing:
                        for m in missing:
                            self._warn_once("ERROR: column/feature '%s' not computed" % m)
                except Exception as e:
                    self._skipped[cond] += 1
                    self._warn("ERROR: could not calc trajectory metrics for oid %s (%s long)\n\t%s" %
                               (oid, n_samples, e))
                    continue

                # provide a nanoseconds after the epoc column (use at your own risk(TM))
                if 'tns' not in df.columns:
                    df['tns'] = np.arange(len(df)) * dt + start_time
                    # should be close too to: df['t_secs'] + 1E-9 * df[t_nsecs]

                r['count'] += 1
                r['start_obj_ids'].append((start_x, start_y, oid, start_framenumber, start_time))
                r['df'].append(df)

                # save uuid
                self._results[cond]['uuids'].append(uuid)

        h5.close()  # maybe this should go in a finally?

        args = vars(args)  # never remember how to use NameSpace
        if not args.get('nocheck', False):
            self._warn('Checking combine health...')
            check_combine_health(self, min_length_f=args.get('lenfilt', None))
            self._warn('Combine seems healthy...')
        else:
            self._warn('Combine health was not checked')

        return uuid


FORMAT_DOCS = """
Exported Data Formats
=====================

Introduction
------------

Files are exported in three different formats csv,pandas dataframe (df)
and matlab (mat).

Depending on the choice of index and the output format the final data should
be interpreted with the following knowledge.

General Concepts
----------------
All data is a combination of that collected from the tracking system
(at precisely 1/frame_rate intervals) and that recorded by the experiment
(at any interval). Frame rate is typically 100 or 120Hz.

The tracking data includes
 * x,y,z (position, m)
 * framenumber
 * tns (time in nanoseconds)
 * vx,vy,vz,velocity (velocity, m/s)
 * ax,ay,az (acceleration, m/s2)
 * theta (heading, rad)
 * dtheta (turn rate, rad/s)
 * radius (distance from origin, m)
 * omega (?)
 * rcurve (radius of curvature of current turn, m)

The experimental data contained in the csv file can include any other fields,
however it is guarenteed to contain at least the following
 * t_sec (unix time seconds component)
 * t_nsec (unix time, sub-second component as nanoseconds)
 * framenumber
 * condition (string)
 * lock_object
 * exp_uuid (string)

** Note **
When the tracking data and the experimental data is combined, any columns that
are identically named in both will be renamed. Tracking data colums that have been
renamed are suffixed with '_h5' while csv columns are added '_csv'

Index Choice
------------
The index choice of the files is denoted by the filename suffix; _framenumber,
_none, _time. According to the choice of index the two sources of data (tracking
and csv) are combined as follows.

** Framenumber index **
The most recent (in time) record for each framenumber is taken from the
experimental (csv) data. If the csv was written at a faster rate than the
tracking data some records will be lost. If the csv was written slower
than the tracking data then there will be missing elements in the columns
of data originating from the csv.

Data with a framenumber index will not contain framenumber column(s) but will
contain tns and t_sec/nsec columns

The framenumber is guarenteed to only increase, but may do so non-monotonically
(due to missing tracking data for example)

** No (none) index **
All records from tracking and experimental data are combined together (temporal
order is preserved). Columns may be missing elements. The wall-clock time for the
tracking data is stored in tns. The wall-clock time for experimental csv
data rows can be reconstructed from t_sec+t_nsec.

Data with no index will contain framenumber columns (with _csv and _h5 suffixes)
and will also contain tns and t_sec/nsec columns

** Time index **
Data with time index is often additionally resampled, indicated by the
file name being timeXXX where X is an integer. If resampled, the string XXX
is defined here -
http://pandas.pydata.org/pandas-docs/dev/timeseries.html#offset-aliases

For example a file named _time10L has been resampled to a 10 millisecond timebase.

This is time aware resampling, so any record from either source that did not
occur at the time instant is resampled. Data is up-sampled by padding the
most recent value forward, and down-sampled by taking the mean over the
interval.

Data with a time index will contain framenumber columns (with _csv and _h5 suffixes)
and tns and t_sec/nsec columns.

If the data has NOT been resampled the data may still contain missing rows

Output Format
-------------
In addition to the colum naming and data combining overview just given,
the following things should be considered when loading exported data. 

** csv files **
The first colum contains the index. If 'framenumber' was chosen the column is
labeled 'framenumber'. If 'none' index was chosen the column is
left unlabeled and the values are monotonically increasing integers. If 'time'
was chosen the column is labeled 'time' and contains strings of the
format '%Y-%m-%d_%H:%M:%S.%f'

** mat files **
The mat files will also contain a variable 'index' which is an integer
for 'framenumber' and 'none' types. If the index type is 'time' then the values
are nanoseconds since unix epoch.

** df files **
Pandas dataframes should contain information as previously described and also
the data types and values for all with full precision

"""


def write_result_dataframe(dest, df, index, to_df=True, to_csv=True, to_mat=True):
    dest = dest + '_' + safe_condition_string(index)

    kwargs = {}
    if index == 'framenumber':
        kwargs['index_label'] = 'framenumber'
    elif index.startswith('time'):
        kwargs['index_label'] = 'time'

    if to_csv:
        df.to_csv(dest + '.csv', **kwargs)

    if to_df:
        df.to_pickle(dest + '.df')

    if to_mat:
        dict_df = df.to_dict('list')
        dict_df['index'] = df.index.values
        scipy.io.savemat(dest + '.mat', dict_df, oned_as='column')

    formats = ('csv' if to_csv else '',
               'df' if to_df else '',
               'mat' if to_mat else '')

    return "%s.{%s}" % (dest, ','.join(filter(len, formats)))

# TODO: whatid and other stuff should be saved to the cache
