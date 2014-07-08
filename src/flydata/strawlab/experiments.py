# coding=utf-8
"""Metadata and trajectories (files) aggregate on FreeflightExperiment objects.
N.B. this is a thin version of that in sandbox; it will be better to reconstruct that functionality from scratch
"""
from itertools import chain

from joblib import Parallel, delayed, cpu_count
from flydata.strawlab.files import FreeflightAnalysisFiles
from flydata.strawlab.metadata import FreeflightExperimentMetadata
from flydata.strawlab.trajectories import FreeflightTrajectory


class FreeflightExperiment(object):

    def __init__(self, uuid, do_cache=False):
        """Freeflight experiment data aggregation (from DB + files) and local caching.

        The main goal of this class is to serve FreeflightTrajectories. Each of these trajectories
        is associated with result-wide metadata and postprocessing information.

        Parameters
        ----------
        uuid : string
            The identifier for this experiment in the strawlab infrastructure
        """

        super(FreeflightExperiment, self).__init__()

        self._uuid = uuid
        self._md = FreeflightExperimentMetadata(uuid=self._uuid)

        # ATM we only extract trajectories from analysis pickle files. There the data is:
        #   - nicely postprocessed by the combine machinery
        #   - hopefully visually inspected by the researcher
        # Also, look at the sandbox
        # (there this class handles also caching, series computation and partial data retrieval)

        # Cache trajectories
        self._trajs_in_memory = None
        if do_cache:
            self.trajectories()

    def uuid(self):
        return self._uuid

    def md(self):
        return self._md

    def num_trajectories(self):
        return len(self.trajectories())

    def conditions(self):
        return sorted(set(traj.condition() for traj in self.trajectories()))

    def sfff(self, filter_id=None):
        return FreeflightAnalysisFiles.from_uuid(self.uuid(), filter_id=filter_id)

    def trajectories(self, filter_id=None, conditions=None):
        """Returns the list of FreeflightTrajectory objects."""
        if self._trajs_in_memory is None:
            self._trajs_in_memory = [FreeflightTrajectory(self.md(), oid, framenumber0, time0, condition, df)
                                     for condition, x0, y0, oid, framenumber0, time0, df in
                                     self.sfff(filter_id=filter_id).trajs(conditions=conditions)]  # N.B. missing dt
        return self._trajs_in_memory


def load_freeflight_experiments(uuids,
                                project_root=None,
                                lazy=True, n_jobs=cpu_count(),
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
                                          (uuid=uuid, do_cache=lazy)  # project_root=project_root
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
                                 with_conditions=None):
    experiments = load_freeflight_experiments(uuids,
                                              project_root=project_root,
                                              lazy=True,
                                              n_jobs=n_jobs,
                                              with_conditions=with_conditions)
    return trajectories_from_experiments(experiments, with_conditions=with_conditions)