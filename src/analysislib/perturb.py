import time
import collections
import numpy as np
import pandas as pd

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')

import strawlab_freeflight_experiments.perturb as sfe_perturb

PerturbationHolder = collections.namedtuple('PerturbationHolder', 'df start_idx end_idx obj_id completed')

def collect_perturbation_traces(combine, args):
    results,dt = combine.get_results()

    perturbations = collections.OrderedDict()           #perturb_obj: {obj_id:Perturbation,...}
    completed_perturbations = {c:[] for c in results}   #condition:(perturb_obj,obj_id,perturbation_length,trajectory_length)
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
                completed = False
                if step_obj.completed_perturbation(tmax) and (lidx > fidx):
                    completed_perturbations[cond].append((step_obj,obj_id,tmax,tmax-df['talign'].min()))
                    completed = True

                df['align'] = np.array(range(len(df)), dtype=int) - fidx

                perturbations[step_obj][obj_id] = PerturbationHolder(df, fidx, lidx, obj_id, completed)

    return perturbations, completed_perturbations, perturbation_conditions

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

