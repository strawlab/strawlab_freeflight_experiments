import roslib
roslib.load_manifest('strawlab_freeflight_experiments')
import analysislib.util

import matplotlib.pyplot as plt
import numpy as np

UUID = "17b2a814ee8b11e4b9316c626d3a008a"

combine = analysislib.util.get_combiner_for_uuid(UUID)
combine.add_feature(column_name='saccade')
combine.add_from_uuid(UUID)

#get the results of one flight (by object id)
df,dt,(x0,y0,obj_id,framenumber0,time0,condition,uuid) = combine.get_one_result(4233, framenumber0=737996)

assert dt == 1/100.0

ax = plt.figure('xy').add_subplot(1,1,1)
ax.plot(df['x'].values,df['y'].values)

vx = np.gradient(df['x']) / dt
vy = np.gradient(df['y']) / dt
theta = np.unwrap(np.arctan2(vy,vx))
dtheta = np.gradient(theta) / dt

f = plt.figure('dthetas', figsize=(18,6))
ax2 = f.add_subplot(1,1,1)
ax2.plot(dtheta,'b-',label='manual')
ax2.plot(df['dtheta'].values,'r-',label='combine')
ax2.legend()

f.savefig('dtheta.png')

plt.show()
