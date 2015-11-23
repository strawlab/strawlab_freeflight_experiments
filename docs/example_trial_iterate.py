import numpy as np
import matplotlib.pyplot as plt

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')
import analysislib.util as autil
import analysislib.combine as acombine
import analysislib.args as aargs

def get_combine(uuids, csv_suffix, **filter_args):
    filter_args['uuid'] = np.atleast_1d(uuids).tolist()

    parser,args = aargs.get_default_args(**filter_args)

    c = acombine.CombineH5WithCSV(csv_suffix=csv_suffix)
    c.add_feature(column_name='radius')
    c.add_feature(column_name='dtheta_deg')
    c.add_feature(column_name='angle_to_post_at_origin_deg')
    c.add_from_args(args)

    return c

def perform_an_operation_on_every_trial(combine):
    results,dt = combine.get_results()

    res = []
    for condition,r in results.iteritems():
        for df,(x0,y0,obj_id,framenumber0,time0) in zip(r['df'], r['start_obj_ids']):
            res.append(df['radius'].mean())

    fig,ax = plt.subplots()
    ax.hist(res)
    ax.set_xlabel('all radius')

if __name__ == "__main__":
    combine = get_combine('13b5593e386711e582c06c626d3a008a',
                               'conflict.csv',
                               arena='flycave',
                               lenfilt=2,
                               vfilt='triminterval',
                               vfilt_min='0.05')

    perform_an_operation_on_every_trial(combine)

    plt.show()

