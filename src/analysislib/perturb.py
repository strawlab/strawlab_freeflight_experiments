import time
import collections
import numpy as np
import pandas as pd

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')

import strawlab_freeflight_experiments.perturb as sfe_perturb

PerturbationHolder = collections.namedtuple('PerturbationHolder', 'df start_idx end_idx obj_id completed start_ratio, perturbation_length, trajectory_length, condition')

def collect_perturbation_traces(combine, args):
    results,dt = combine.get_results()

    perturbations = collections.OrderedDict()           #perturb_obj: {obj_id:PerturbationHolder,...}
    perturbation_conditions = {}                        #perturb_obj: cond

    for cond in sorted(results):

        perturb_desc = cond.split("/")[-1]

        pklass = sfe_perturb.get_perturb_class(perturb_desc)

        #only plot perturbations
        if pklass == sfe_perturb.NoPerturb:
            continue

        step_obj = pklass(perturb_desc)
        perturbations[step_obj] = {}

        perturbation_conditions[step_obj] = cond

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

                #ensure we get a unique obj_id for later grouping. That is not necessarily
                #guarenteed because obj_ids may be in multiple conditions, so if need be
                #create a new one
                if obj_id in perturbations[step_obj]:
                    obj_id = int(time.time()*1e6)
                df['obj_id'] = obj_id

                t = time0 + (np.arange(0,len(df),dtype=float) * dt)
                df['time'] = t
                df['talign'] = t - t[fidx]

                tmax = df['talign'].max()
                traj_length = tmax - df['talign'].min()

                completed = step_obj.completed_perturbation(tmax) and (lidx > fidx)

                df['align'] = np.array(range(len(df)), dtype=int) - fidx

                #save both the exact value, and an identifier to signify which
                #part of the arena the perturbation started in. The identifier
                #is based on the ratio range_funcs, range_chunks magic. A range string
                #0.4|0.46|0.5|0.8 is the pairs of ranges (0.4->0.46) and (0.5->0.8).
                #if the fly started its perturbation with a ratio of 0.55 the range
                #chunk identifier is 1 (the 2nd pair in the range string).
                start_ratio = df.iloc[fidx]['ratio']
                df['ratio_range_start_id'] = step_obj.get_perturb_range_identifier(start_ratio)

                ph_obj = PerturbationHolder(df, fidx, lidx, obj_id, completed, start_ratio, tmax, traj_length, cond)

                perturbations[step_obj][obj_id] = ph_obj

    return perturbations, perturbation_conditions

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




