import collections

import numpy as np
import pandas as pd

import roslib

roslib.load_manifest('strawlab_freeflight_experiments')

import strawlab_freeflight_experiments.perturb as sfe_perturb

# uuid, obj_id, start_frame: trial id
# perturb_id: placeholder in case there can be more than one perturbation per trial
# condition: the condition string
# start_criteria, start_reason, start_id: the type, why and where the perturbation started
# start_idx, end_idx: interval where the perturbation took place (half-closed [start_idx, end_idx), ala python)
# completed, completed_pct: is the perturbation considered to be complete and which percentage was completed?
# perturbation_length, trajectory_length: lengths, in seconds
# df: the series dataframe, may or may not contain alignment columns
PerturbationHolder = collections.namedtuple('PerturbationHolder',
                                            'uuid, obj_id, start_frame, '
                                            'perturb_id, '
                                            'condition, '
                                            'start_criteria, start_reason, start_id, '
                                            'start_idx, end_idx, '
                                            'completed, completed_pct, '
                                            'perturbation_length, trajectory_length, '
                                            'df')


def find_step_obj(cond, condition_conf=None):
    """Returns the perturber used during the condition or None if cond it is not a perturbed condition."""

    if condition_conf:
        try:
            step_obj = sfe_perturb.get_perturb_object_from_condition(condition_conf)
        except KeyError:
            return None
    else:
        # backwards compatibility for old pre-yaml experiments where the perturb_descripor
        # was assumed to be the last element in the condition string
        perturb_desc = cond.split("/")[-1]
        step_obj = sfe_perturb.get_perturb_object(perturb_desc)

        # add a sanity check as we want to improve the degree we trust ConditionCompat
        # to do the backwards compatibility for us
        if isinstance(step_obj, sfe_perturb.NoPerturb):
            print "WARNING: WE THOUGHT %s WAS A PERTURBATION BUT IT IS NOT. UPDATE ConditionCompat.PERTURB_RE" % cond

    return None if isinstance(step_obj, sfe_perturb.NoPerturb) else step_obj


def extract_perturbation_interval(df, step_obj, dt, completion_threshold=0.98):
    """Returns a 6-tuple start_index, last_index, perturbation_length, traj_length, completed, completed_pct."""
    # recall that pandas fillna is really slow
    perturb_progress = df['perturb_progress'].fillna(method='ffill').fillna(method='bfill')
    # find the start of the perturbation (where perturb_progress == 0)
    # is 0 guaranteed bo be there whenever there is perturbation data?
    # is_perturbed = df['perturb_progress'] >= 0
    perturb_start = np.where(perturb_progress.values == 0)[0]
    if 0 == len(perturb_start):
        return (None,) * 6
    first_idx = perturb_start[0]
    # find the end of the perturbation (where perturb_progress is max)
    # this is what it was before:
    #   last_idx = df['perturb_progress'].argmax()
    # is this what we want in case perturb_progress max happens more than once? or should we get the last occurrence?
    last_idx = np.where(perturb_progress == perturb_progress.max())[0][-1] + 1  # +1 => python interval

    # Perturbation length and completion
    traj_length = (len(df) - 1) * dt
    perturbation_length = (last_idx - first_idx - 1) * dt
    completed = step_obj.completed_perturbation(perturbation_length, completion_threshold) and (last_idx > first_idx)
    completed_pct = step_obj.completed_perturbation_pct(perturbation_length)

    return first_idx, last_idx, perturbation_length, traj_length, completed, completed_pct


def extract_perturbations(df, uuid, obj_id, framenumber0, cond, time0, dt,
                          step_obj,
                          completion_threshold=0.98,
                          save_alignment=True):
    """Returns a list of PerturbationHolder objects with the perturbations for a single trial.

    Parameters
    ----------
    df: pandas DataFrame with the trial time series

    uuid, obj_id, framenumber0: the trial identifier

    cond: the condition string

    time0: time of the first trial observation (s)

    dt: sampling period (s)

    step_obj: Perturber instance

    completion_threshold: float in [0, 1], default 0.98
      the percentage at which a perturbation is considered to be "completed"

    save_alignment: boolean, default True
      if True, df is copied and 3 new perturbation alignment columns are added:
        - time: a time index, based on time0 and dt
        - talign: time - time[perturbation_start]
        - align: a 0-based frame-align index,
                 0 is the perturbation start, negative numbers indicate pre-perturbation

    Returns
    -------
    A list of PerturbationHolder objects (can only be singleton ATM), empty if no perturbation can be found.
    """

    # TODO: would we like to make this a function of the perturber?

    first_idx, last_idx, perturbation_length, traj_length, completed, completed_pct = \
        extract_perturbation_interval(df, step_obj, dt, completion_threshold)

    if first_idx is None:
        return []

    # Why and where the perturbation started?
    # We probably would like to abstract this to something like "start_reason"
    # (for example, now start can be time-triggered)
    start_id = None
    start_ratio = None
    if step_obj.criteria_type == sfe_perturb.Perturber.CRITERIA_TYPE_RATIO:
        # save both the exact value, and an identifier to signify which
        # part of the arena the perturbation started in. The identifier
        # is based on the ratio range_funcs, range_chunks magic. A range string
        # 0.4|0.46|0.5|0.8 is the pairs of ranges (0.4->0.46) and (0.5->0.8).
        # if the fly started its perturbation with a ratio of 0.55 the range
        # chunk identifier is 1 (the 2nd pair in the range string).
        ratio = df['ratio'].fillna(method='ffill').fillna(method='bfill')
        start_ratio = ratio.iloc[first_idx]
        start_id = step_obj.get_perturb_range_identifier(start_ratio)

    # Time and frame alignment series
    if save_alignment:
        df = df.copy(deep=True)  # this is what was happening before, we might want to allow not-copying
        df['time'] = time0 + (np.arange(len(df), dtype=float) * dt)
        df['talign'] = df['time'] - df['time'].iloc[first_idx]
        df['align'] = np.arange(len(df), dtype=int) - first_idx
        # Are we using all these already? They are easy to create when needed...
        # Copying the dataframe is very costly; we might want to just return a dataframe with these alignment columns...

    return [PerturbationHolder(uuid=uuid, obj_id=obj_id, start_frame=framenumber0,
                               perturb_id=0,
                               condition=cond,
                               start_criteria=step_obj.criteria_type, start_reason=start_ratio, start_id=start_id,
                               start_idx=first_idx, end_idx=last_idx,
                               completed=completed, completed_pct=completed_pct,
                               perturbation_length=perturbation_length, trajectory_length=traj_length,
                               df=df)]


def collect_perturbation_traces(combine,
                                as_dictionary=True,
                                allowed_perturbation_types=None,
                                extractor=extract_perturbations,
                                **extractor_params):
    """
    Returns a two-tuple (perturbations, perturbation_objects) with the perturbations in the combine object.
     - perturbations is either a dictionary {condition: [PerturbationHolder]} or a list of PerturbationHolder objects.
     - perturbation_objects is a dictionary {condition: Perturber}

    Parameters
    ----------
    combine: Combine object

    as_dictionary: boolean, default True
      if True, perturbations is a dictionary; if False, perturbations is a list

    allowed_perturbation_types: list of strings (or None-like), default None
      if supplied, a list of perturbation types to analyse ('idinput', 'step', for example)      

    extractor: a function with the same signature as extract_perturbation, returning a list of perturbations per trial
      see the extract_perturbations method

    extractor_params: keyword arguments passed to "extractor" on each trial
    """

    results, dt = combine.get_results()

    perturbations = []          # list of PerturbationHolder objects
    perturbation_objects = {}   # cond: perturb_obj

    for cond in sorted(results):

        condition_obj = combine.get_condition_object(cond)
        if not condition_obj.is_type('perturbation'):
            continue

        condconf = combine.get_condition_configuration(combine.get_condition_name(cond))
        step_obj = find_step_obj(cond, condconf)

        if step_obj is None:
            continue

        if allowed_perturbation_types and (step_obj.NAME not in allowed_perturbation_types):
            continue

        perturbation_objects[cond] = step_obj

        r = results[cond]

        # Work even if no uuids are present
        uuids = r['uuids'] if 'uuids' in r else []
        if len(uuids) != len(r['df']):
            uuids = [None] * len(r['df'])

        for df, (x0, y0, obj_id, framenumber0, time0), uuid in zip(r['df'], r['start_obj_ids'], uuids):
            perturbations.extend(extractor(df,
                                           uuid, obj_id, framenumber0,
                                           cond,
                                           time0, dt,
                                           step_obj,
                                           **extractor_params))

    if as_dictionary:
        perturbations_dict = {}
        for ph in perturbations:
            if ph.condition not in perturbations_dict:
                perturbations_dict[ph.condition] = []
            perturbations_dict[ph.condition].append(ph)
        return perturbations_dict, perturbation_objects

    return perturbations, perturbation_objects


def get_input_output_columns(step_obj):
    # For SID. Shouldn't this be part of step_obj?
    if step_obj.what == 'rotation_rate':
        system_y = 'dtheta'
        system_u = 'rotation_rate'
    elif step_obj.what == 'z':
        system_y = 'vz'
        system_u = 'v_offset_rate'
    else:
        raise Exception("Not supported: %s" % step_obj.what)

    return system_u, system_y


def perturbations2df(perturbations, short_names=True, keep_df=True):
    """Returns a dataframe view of the perturbations list."""
    if perturbations is None or 0 == len(perturbations):
        return None
    df = pd.DataFrame(perturbations, columns=perturbations[0]._fields)
    if short_names:
        df = df.rename(columns={'start_frame': 'startf',
                                'obj_id': 'oid'})
    if not keep_df:
        del df['df']
    return df
