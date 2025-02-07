import numpy as np
import matplotlib.pyplot as plt

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')
import analysislib.util as autil
import analysislib.combine as acombine
import analysislib.args as aargs

def get_combine_dataframe(uuids, csv_suffix, **filter_args):
    filter_args['uuid'] = np.atleast_1d(uuids).tolist()

    parser,args = aargs.get_default_args(**filter_args)

    c = acombine.CombineH5WithCSV(csv_suffix=csv_suffix)
    c.add_feature(column_name='radius')
    c.add_feature(column_name='dtheta_deg')
    c.add_feature(column_name='angle_to_post_at_origin_deg')
    c.add_from_args(args)

    aargs.describe(c,args)

    df = c.get_observations_dataframe(cols=('radius', 'dtheta_deg', 'angle_to_post_at_origin_deg'))

    return df,c

def perform_an_operation_on_every_trial(combine, df):

    res = []
    for trial,_df in df.groupby(['uuid','obj_id','framenumber0']):
        uuid,obj_id,framenumber0 = trial
        #condition = _df['condition'].unique()[0]
        res.append(_df['radius'].mean())

    fig,ax = plt.subplots()
    ax.hist(res)
    ax.set_xlabel('all radius')

def perform_an_operation_on_every_condition(combine, df):

    res = {}
    for condition,_df in df.groupby('condition'):
        print combine.get_condition_object(condition)
        res[combine.get_condition_name(condition)] = _df['radius'].mean()

    print res

if __name__ == "__main__":
    df,combine = get_combine_dataframe('13b5593e386711e582c06c626d3a008a',
                               'conflict.csv',
                               arena='flycave',
                               lenfilt=2,
                               vfilt='triminterval',
                               vfilt_min='0.05')

    for cond,_df in df.groupby('condition'):
        fig,ax = plt.subplots()
        fig.suptitle(cond)
        ax.hist(_df['radius'].values)

    perform_an_operation_on_every_trial(combine,df)
    perform_an_operation_on_every_condition(combine,df)

    plt.show()

