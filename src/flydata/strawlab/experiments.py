# coding=utf-8
"""Metadata and trajectories (files) aggregate on FreeflightExperiment objects.
N.B. this is a thin version of that in sandbox; it will be better to reconstruct that functionality from scratch
"""
from itertools import chain
import os.path as op
import cPickle as pickle
from time import time

from joblib import Parallel, delayed, cpu_count
from flydata.misc import ensure_dir
from flydata.strawlab.contracts import check_contracts, NoMissingValuesContract, NoHolesContract, AllNumericContract

from flydata.strawlab.files import FreeflightAnalysisFiles
from flydata.strawlab.metadata import FreeflightExperimentMetadata
from flydata.strawlab.trajectories import FreeflightTrajectory


class FreeflightExperiment(object):

    def __init__(self,
                 uuid,
                 lazy=True,
                 cache_root_dir=None,
                 md_transformers=None,
                 traj_transformers=None):
        """Freeflight experiment data aggregation (from DB + files) and local caching.

        The main goal of this class is to serve FreeflightTrajectories. Each of these trajectories
        is associated with result-wide metadata and postprocessing information.

        Parameters
        ----------
        uuid : string
            The identifier for this experiment in the strawlab infrastructure
        """

        super(FreeflightExperiment, self).__init__()

        # Manage caching
        self.cache_root = cache_root_dir

        self._uuid = uuid

        # Experiment metadata
        self._md = FreeflightExperimentMetadata(uuid=self._uuid, json_file_or_dir=self.cache_root)
        if md_transformers is not None:
            self._md.apply_transform_inplace(**md_transformers)
        if self.cache_root is not None:
            self._md.genotype()  # Force populate
            self._md.to_json_file()  # FIXME: do not write if it is already there

        # ATM we only extract trajectories from analysis pickle files. There the data is:
        #   - nicely postprocessed by the combine machinery
        #   - hopefully visually inspected by the researcher
        # Also, look at the sandbox
        # (there this class handles also caching, series computation and partial data retrieval)

        # Cache trajectories
        self._trajs_in_memory = None
        self._traj_transformers = traj_transformers
        if not lazy:
            self.trajectories()

    def uuid(self):
        return self._uuid

    def md(self):
        return self._md

    def num_trajectories(self):
        return len(self.trajectories())

    def conditions(self):
        return sorted(set(traj.condition() for traj in self.trajectories()))  # Arguably, should be read from metadata

    def sfff(self, filter_id=None):
        if self.cache_root is None:
            return FreeflightAnalysisFiles.from_uuid(self.uuid(), filter_id=filter_id)
        return FreeflightAnalysisFiles.from_path(op.join(self.cache_root, self.uuid()))
        # FIXME: improve API and document

    def trajectories(self, filter_id=None, conditions=None):
        """Returns the list of FreeflightTrajectory objects."""
        if self._trajs_in_memory is None:
            self._trajs_in_memory = [FreeflightTrajectory(self.md(), oid, framenumber0, time0, condition, df)
                                     for condition, x0, y0, oid, framenumber0, time0, df in
                                     self.sfff(filter_id=filter_id).trajs(conditions=conditions)]  # N.B. missing dt
            if self._traj_transformers is not None:
                for traj in self._trajs_in_memory:
                    traj.apply_transform_inplace(**self._traj_transformers)
        return self._trajs_in_memory


def load_freeflight_experiments(uuids,
                                project_root=None,
                                lazy=True,
                                n_jobs=cpu_count(),
                                md_transforms=None,
                                traj_transforms=None,  # This is a harsh API...
                                with_conditions=None):
    """Loads all the experiment into memory, possibly caching to local disk on the way.

    Parameters
    ----------
    uuids: string list
        The uuids identifying the experiments

    project_root: path
        The directory where the analysis data and results will be stored

    lazy: boolean, default False
        If False, all the trajectories will be loaded in RAM; otherwise, lazy loading

    n_jobs: int, defaults to taking all CPUs
        The number of processes used to cache and load the data into memory

    with_conditions: string list, default None
        If a non-empty list of strings, return only experiments containing the conditions specified in the list.
        If None, return all experiments regardless of the conditions they contain.

    Returns
    -------
    A FreeflightExperimentAggregator list with the loaded and kept experiments
    """
    experiments = Parallel(n_jobs=n_jobs)(delayed(FreeflightExperiment)
                                          (uuid=uuid,
                                           lazy=lazy,
                                           md_transformers=md_transforms,
                                           traj_transformers=traj_transforms)  # project_root=project_root
                                          for uuid in uuids)
    # Get rid of experiments without the requested conditions
    if with_conditions is not None:
        with_conditions = set(with_conditions)
        experiments = filter(lambda experiment: len(with_conditions & set(experiment.conditions())) > 0, experiments)
    return experiments


# ah pickle limits
def _trajs_from_experiment(experiment):
    return experiment.trajectories()


def trajectories_from_experiments(experiments,
                                  project_root=None,
                                  n_jobs=1,
                                  with_conditions=None):
    # Read to memory
    trajs = Parallel(n_jobs=n_jobs)(delayed(_trajs_from_experiment)
                                    (experiment)
                                    for experiment in experiments)
    # flatten
    trajs = list(chain(*trajs))
    # filter by condition
    if with_conditions is not None:
        with_conditions = set(with_conditions)
        return filter(lambda traj: traj.condition() in with_conditions, trajs)
    return trajs


def load_freeflight_trajectories(uuids,
                                 project_root=None,
                                 n_jobs=1,
                                 md_transforms=None,
                                 traj_transforms=None,
                                 with_conditions=None):
    """
    Parameters
    ----------
    uuids: string or iterator over strings
      The uuid(s) of the experiments we want to load

    project_root: string, default None
      Path to the directory where the experiments are stored.
      If None, read from the global freeflight data repository

    n_jobs: int or None, default 1
      Number of threads used to read the data in.
      If None or less than 1, use all CPUs.
      If 1, no multithreading is used (useful to debug).

    md_transforms: dictionary {'string': (x) -> y}, default None
      Transformations to apply to the experiment metadata fields.
      For example: md_transforms={'genotype': lambda genotype: 'hardcoded-genotype'}

    traj_transforms:
      Transformations to apply to the trajectories metadata fields.
      For example: traj_transforms={'condition': lambda genotype: 'hardcoded-condition'}

    with_conditions: string list or None
      Only trajectories corresponding to trials with these conditions will be kept.
      If None, all trajectories are kept.

    Returns
    -------
    A list of freeflight trajectories, sorted by experiment and transformed according to these transforms.
    """
    if isinstance(uuids, (unicode, basestring)):
        uuids = [uuids]
    if n_jobs is None or n_jobs < 1:
        n_jobs = cpu_count()
    experiments = load_freeflight_experiments(uuids,
                                              project_root=project_root,
                                              lazy=True,
                                              n_jobs=n_jobs,
                                              with_conditions=with_conditions,
                                              md_transforms=md_transforms,
                                              traj_transforms=traj_transforms)
    return trajectories_from_experiments(experiments, with_conditions=with_conditions)


# These are contracts for the data in our trajectories
DEFAULT_DATA_CONTRACTS = (
    NoMissingValuesContract(),  # No missing values, please
    NoHolesContract(),          # No holes in time series, please
    AllNumericContract(),       # No "object" columns, please
)


def read_preprocess_cache_data(uuids,
                               exp_reader=load_freeflight_experiments,
                               traj_reader=load_freeflight_trajectories,
                               cache_dir=None,
                               data_name='prepared_data',
                               mirror_strawscience_locally=False,
                               transformers=(),
                               contracts=DEFAULT_DATA_CONTRACTS,
                               series_extractors=(),
                               recompute=False):
    """
    Freeflight-data-analysis trajectories reading, preprocessing, checking and extending example,
    with provenance checking.

    Parameters
    ----------
    uuids: string or string list
      The ids for the experiment we want to retrieve.

    cache_dir: string
      Path to a local directory where we will store the retrieved data (and some results)

    data_name: string
      An identifier for the data that will come out of this function

    mirror_strawscience_locally: boolean, default False
      If true, the analysis files from strawscience are copied to cache_dir
      (useful for bringing the original data with you anywhere)

    transformers: iterator of Transformer-like objects
      These will transform our trajectories after loading (e.g. missing-value treatment)

    contracts: iterator of DataContract-like objects
      These will check that our trajectories obey certain rules (e.g. all values are numeric)

    series_extractors: iterator of SeriesExtractor-like object
      These will derive new time series from the existing data (e.g. compute instantaneous direction towards a post)

    recompute: boolean, default False
      We store all these computations in a cache in disk;
      if this is True, we will recompute everything even if the cache already exists

    Returns
    -------
    A tuple (trajs, provenance).
      - trajs: list of FreeflightTrajectory objects, carrying metadata and time-series for a trial.
      - provenance: list of operation-identifiers with all the history of data-processing for these trajectories
    """

    # We will store the results of this function here...
    cache_file = op.join(cache_dir, '%s.pkl' % data_name)

    # Provenance: store what we do to the data
    provenance = [FreeflightExperiment(uuid=uuids[0]).sfff().what().id()]  # Assume all same, lame...
    provenance_file = op.join(cache_dir, '%s-provenance.log' % data_name)

    if recompute or not op.isfile(cache_file):

        # Local mirror of combined trajectories and metadata
        if mirror_strawscience_locally:
            for exp in exp_reader(uuids):
                exp.sfff().mirror_to(op.join(cache_dir, exp.uuid()))

        # Load the trajectories from the local cache
        print '\tLoading data from original combine results...'
        start = time()
        trajs = traj_reader(uuids=uuids, project_root=cache_dir)  # WARNING missing provenance from traj_reader
        print '\tLoaded %d trajectories, it took %.2f seconds' % (len(trajs), time() - start)

        # Apply transformers
        print '\tTransforming the trajectories data-set...'
        start = time()
        for transformer in transformers:
            print '\t\t' + transformer.what().id()
            trajs = transformer.fit_transform(trajs)
            provenance.append(transformer.what().id())
        print '\tTransformations took %.2f seconds, there are %d trajectories left' % (time() - start, len(trajs))

        # Check contracts
        print '\tChecking data contracts...'
        start = time()
        checked = check_contracts(trajs, contracts)
        provenance.extend([contract.what().id() for contract in contracts])
        print '\tChecked:\n\t\t%s' % '\n\t\t'.join(checked)
        print '\tCheck contracts took %.2f seconds' % (time() - start)

        # Compute some extra time-series
        print '\tComputing more time-series...'
        start = time()
        for series_extractor in series_extractors:
            print '\t\t' + series_extractor.what().id()
            series_extractor.compute(trajs)
            provenance.append(series_extractor.what().id())
        print 'Compute some more series took %.2f seconds' % (time() - start)
        # N.B. we should check more contracts after this, but careful with some missing values that are valid...
        #      that's because we do not support (yet) variable length series

        # Save to our extraordinary data format (a pickle)
        ensure_dir(cache_dir)
        with open(cache_file, 'wb') as writer:
            pickle.dump(trajs, writer, protocol=pickle.HIGHEST_PROTOCOL)

        # Save provenance information
        with open(provenance_file, 'w') as writer:
            writer.write('\n'.join(provenance))

    with open(cache_file) as reader_trajs, open(provenance_file) as reader_provenance:
        trajs = pickle.load(reader_trajs)
        provenance = map(str.strip, reader_provenance.readlines())
        return trajs, provenance
