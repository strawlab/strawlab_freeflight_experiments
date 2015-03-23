import time
import collections
import numpy as np
import pandas as pd

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')

import strawlab_freeflight_experiments.perturb as sfe_perturb

PerturbationHolder = collections.namedtuple('PerturbationHolder', 'df start_idx end_idx obj_id completed completed_pct start_ratio, perturbation_length, trajectory_length, condition')

def collect_perturbation_traces(combine, completion_threshold=0.98):
    results,dt = combine.get_results()

    perturbations = collections.OrderedDict()           #cond: {obj_id:PerturbationHolder,...}
    perturbation_objects = {}                           #cond: perturb_obj

    for cond in sorted(results):

        condition_obj = combine.get_condition_object(cond)
        if not condition_obj.is_type('perturbation'):
            continue

        condition_conf = combine.get_condition_configuration(combine.get_condition_name(cond))
        if condition_conf:
            try:
                step_obj = sfe_perturb.get_perturb_object_from_condition(condition_conf)
            except KeyError:
                #new style yaml experiment, not a perturbation condition
                continue
        else:
            #backwards compatibility for old pre-yaml experiments where the perturb_descripor
            #was assumed to be the last element in the condition string
            perturb_desc = cond.split("/")[-1]
            step_obj = sfe_perturb.get_perturb_object(perturb_desc)

            #add a sanity check as we want to improve the degree we trust ConditionCompat
            #to do the backwards compatibility for us
            if isinstance(step_obj,sfe_perturb.NoPerturb):
                print "WARNING: WE THOUGHT %s WAS A PERTURBATION BUT IT IS NOT. UPDATE ConditionCompat.PERTURB_RE" % cond

        #only plot perturbations
        if isinstance(step_obj,sfe_perturb.NoPerturb):
            continue

        perturbations[cond] = []
        perturbation_objects[cond] = step_obj

        r = results[cond]

        for _df,(x0,y0,obj_id,framenumber0,time0) in zip(r['df'], r['start_obj_ids']):

            df = _df.fillna(method='ffill')
            df['condition'] = cond

            #find the start of the perturbation (where perturb_progress == 0)
            z = np.where(df['perturb_progress'].values == 0)
            if len(z[0]):
                fidx = z[0][0]

                #find the index of the last perturbation (max -1)
                l = np.where(df['perturb_progress'].values == df['perturb_progress'].max())
                lidx = l[0][0]

                #FIXME: obj_id might not be unique here if later grouping occurs.
                #should use a strong uuid as per Santi
                df['obj_id'] = obj_id

                t = time0 + (np.arange(0,len(df),dtype=float) * dt)
                df['time'] = t
                df['talign'] = t - t[fidx]

                tmax = df['talign'].max()
                traj_length = tmax - df['talign'].min()

                completed = step_obj.completed_perturbation(tmax, completion_threshold) and (lidx > fidx)
                completed_pct = step_obj.completed_perturbation_pct(tmax, completion_threshold)

                df['align'] = np.array(range(len(df)), dtype=int) - fidx

                #save both the exact value, and an identifier to signify which
                #part of the arena the perturbation started in. The identifier
                #is based on the ratio range_funcs, range_chunks magic. A range string
                #0.4|0.46|0.5|0.8 is the pairs of ranges (0.4->0.46) and (0.5->0.8).
                #if the fly started its perturbation with a ratio of 0.55 the range
                #chunk identifier is 1 (the 2nd pair in the range string).
                start_ratio = df.iloc[fidx]['ratio']
                start_id = step_obj.get_perturb_range_identifier(start_ratio)

                df['ratio_range_start_id'] = start_id

                ph_obj = PerturbationHolder(df, fidx, lidx, obj_id, completed, completed_pct, start_ratio, tmax, traj_length, cond)
                perturbations[cond].append(ph_obj)

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




