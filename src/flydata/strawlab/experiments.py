# coding=utf-8
"""Metadata and trajectories (files) aggregate on FreeflightExperiment objects.
N.B. this is a thin version of that in sandbox; it will be better to reconstruct that functionality from scratch
"""
from itertools import chain
import os.path as op

from joblib import Parallel, delayed, cpu_count

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
                                  n_jobs=cpu_count(),
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
                                 n_jobs=cpu_count(),
                                 md_transforms=None,
                                 traj_transforms=None,
                                 with_conditions=None):
    experiments = load_freeflight_experiments(uuids,
                                              project_root=project_root,
                                              lazy=True,
                                              n_jobs=n_jobs,
                                              with_conditions=with_conditions,
                                              md_transforms=md_transforms,
                                              traj_transforms=traj_transforms)
    return trajectories_from_experiments(experiments, with_conditions=with_conditions)