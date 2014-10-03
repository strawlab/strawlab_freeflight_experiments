# coding=utf-8
from flydata.strawlab.experiments import FreeflightExperiment


THINGS_WORKING_AWESOME = (
    # '9a74058e375911e4ac7360a44c2451e5',
    '4ea24fbc375411e4ac7360a44c2451e5',
    '99a31382350311e4be8860a44c2451e5',
    'c94ee180351011e4aae860a44c2451e5',
)


def link_to_experiment(uuid):
    return 'http://strawcore:8080/experiment/fishvr_fishtrax/%s' % uuid

for uuid in THINGS_WORKING_AWESOME:
    print link_to_experiment(uuid)

experiments = [FreeflightExperiment(uuid=uuid) for uuid in THINGS_WORKING_AWESOME]

for exp in experiments:
    print exp.uuid(), exp.md().description(), link_to_experiment(exp.uuid())

import pandas as pd
from collections import defaultdict

df = defaultdict(list)

for exp in experiments:
    df['uuid'].append(exp.uuid())
    df['link'].append(link_to_experiment(exp.uuid()))
    df['comments'].append(exp.md().description())
    df['genotype'].append(exp.md().genotype())

df = pd.DataFrame(data=df)
df.to_csv('/home/santi/max.csv')

df2 = pd.read_csv('/home/santi/max.csv')

df_from_xls = pd.read_excel('/home/santi/max.xls')
experiments = [FreeflightExperiment(uuid=row['uuid']) for _, row in df_from_xls.iterrows()]
for exp in experiments:
    try:
        trajs = exp.trajectories()
        print 'there are %d trajectories and the longest is %d observations' % \
              (len(trajs), max([len(traj.series()) for traj in trajs]))
    except:
        pass