import roslib
roslib.load_manifest('strawlab_freeflight_experiments')
import analysislib.util

combine = analysislib.util.get_combiner("rotation.csv")
combine.add_from_uuid("d21b4ff80c0611e3bfcc6c626d3a008a")

#get the results of one flight (by object id)
df,dt,(x0,y0,obj_id,framenumber0,time0) = combine.get_one_result(4984)

#get all results (se the docs for the structure of the results dictionary)
results, dt = combine.get_results()
