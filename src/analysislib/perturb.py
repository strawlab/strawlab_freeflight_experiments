import collections

import numpy as np
import pandas as pd

import roslib

roslib.load_manifest('strawlab_freeflight_experiments')

import strawlab_freeflight_experiments.perturb as sfe_perturb

PerturbationHolder = collections.namedtuple('PerturbationHolder',
                                            'uuid, obj_id, start_frame, '
                                            'perturb_id, '
                                            'condition, '
                                            'start_ratio, '
                                            'start_idx, end_idx, '
                                            'completed, completed_pct, '
                                            'perturbation_length, trajectory_length, '
                                            'df')


def find_step_obj(cond, condition_conf=None):

    if condition_conf:
        try:
            step_obj = sfe_perturb.get_perturb_object_from_condition(condition_conf)
        except KeyError:
            # new style yaml experiment, not a perturbation condition
            return None
    else:
        # backwards compatibility for old pre-yaml experiments where the perturb_descripor
        # was assumed to be the last element in the condition string
        perturb_desc = cond.split("/")[-1]
        step_obj = sfe_perturb.get_perturb_object(perturb_desc)

    return None if isinstance(step_obj, sfe_perturb.NoPerturb) else step_obj


def extract_perturbations(df, uuid, obj_id, framenumber0, cond, time0, dt,
                          step_obj, completion_threshold=0.98):
    """Returns a list with the perturbations for a single trial."""

    df = df.fillna(method='ffill')

    # find the start of the perturbation (where perturb_progress == 0)
    z = np.where(df['perturb_progress'].values == 0)
    if len(z[0]):
        fidx = z[0][0]

        # find the index of the last perturbation (max -1)
        l = np.where(df['perturb_progress'].values == df['perturb_progress'].max())
        lidx = l[0][0]

        t = time0 + (np.arange(0, len(df), dtype=float) * dt)
        df['time'] = t
        df['talign'] = t - t[fidx]

        tmax = df['talign'].max()
        traj_length = tmax - df['talign'].min()

        completed = step_obj.completed_perturbation(tmax, completion_threshold) and (lidx > fidx)
        completed_pct = step_obj.completed_perturbation_pct(tmax, completion_threshold)

        df['align'] = np.array(range(len(df)), dtype=int) - fidx

        # save both the exact value, and an identifier to signify which
        # part of the arena the perturbation started in. The identifier
        # is based on the ratio range_funcs, range_chunks magic. A range string
        # 0.4|0.46|0.5|0.8 is the pairs of ranges (0.4->0.46) and (0.5->0.8).
        # if the fly started its perturbation with a ratio of 0.55 the range
        # chunk identifier is 1 (the 2nd pair in the range string).
        start_ratio = df.iloc[fidx]['ratio']
        start_id = step_obj.get_perturb_range_identifier(start_ratio)

        df['ratio_range_start_id'] = start_id

        ph_obj = PerturbationHolder(uuid=uuid, obj_id=obj_id, start_frame=framenumber0,
                                    perturb_id=0,
                                    condition=cond,
                                    start_ratio=start_ratio,
                                    start_idx=fidx, end_idx=lidx,
                                    completed=completed, completed_pct=completed_pct,
                                    perturbation_length=tmax, trajectory_length=traj_length,
                                    df=df)
        return [ph_obj]

    return []


def collect_perturbation_traces(combine, completion_threshold=0.98):

    results, dt = combine.get_results()

    perturbations = []          # list of PerturbationHolder objects
    perturbation_objects = {}   # cond: perturb_obj

    for cond in sorted(results):

        condconf = combine.get_condition_configuration(combine.get_condition_name(cond))
        step_obj = find_step_obj(cond, condconf)

        if step_obj is None:
            continue

        perturbation_objects[cond] = step_obj

        r = results[cond]

        # Work even if no uuids are present
        uuids = r['uuids'] if 'uuids' in r else []
        if len(uuids) != len(r['df']):
            uuids = [None] * len(r['df'])

        for df, (x0, y0, obj_id, framenumber0, time0), uuid in zip(r['df'], r['start_obj_ids'], uuids):
            perturbations.extend(extract_perturbations(df,
                                                       uuid,
                                                       obj_id,
                                                       framenumber0,
                                                       cond,
                                                       time0,
                                                       dt,
                                                       step_obj,
                                                       completion_threshold))

    return perturbations, perturbation_objects


def get_input_output_columns(step_obj):
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
    if perturbations is None or 0 == len(perturbations):
        return None
    df = pd.DataFrame(perturbations, columns=perturbations[0]._fields)
    if short_names:
        df = df.rename(columns={'start_frame': 'startf',
                                'obj_id': 'oid'})
    if not keep_df:
        del df['df']
    return df
