import roslib
roslib.load_manifest('strawlab_freeflight_experiments')
import analysislib.util
import analysislib.arenas
import analysislib.plots

import numpy as np
import matplotlib.pyplot as plt

# used for colorline


if __name__ == "__main__":


    UUID = "9b97392ebb1611e2a7e46c626d3a008a"

    combine = analysislib.util.get_combiner_for_uuid(UUID)
    combine.add_feature(column_name='velocity')
    combine.add_feature(column_name='theta')
    combine.add_from_uuid(UUID)

    arena = analysislib.arenas.get_arena('flycave')

    df,dt,(x0,y0,obj_id,framenumber0,time0,condition,uuid) = combine.get_one_result(9)
    d = df.head(1450)

    f = plt.figure()
    ax = f.add_subplot(1,1,1)

    lc = analysislib.plots.colorline(ax,d['x'].values,d['y'].values,d['velocity'].values)
    arena.plot_mpl_line_2d(ax)
    cbar = f.colorbar(lc)
    cbar.set_label('velocity')

    f = plt.figure()
    ax = f.add_subplot(1,1,1)

    pc = analysislib.plots.colorline_with_heading(ax,d['x'].values,d['y'].values,d['velocity'].values,df['theta'].values,deg=False,size_radius=0.05,nskip=70)
    arena.plot_mpl_line_2d(ax)
    ax.plot(d['x'].values,d['y'].values,'k')
    ax.plot(d['x'].values[0],d['y'].values[0],'k8',ms=8,label='start')
    ax.plot(d['x'].values[-1],d['y'].values[-1],'ks',ms=8,label='end')
    ax.legend()
    f.colorbar(pc)

    plt.show()
