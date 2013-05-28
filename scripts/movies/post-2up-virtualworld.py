import tables
import numpy as np
import scipy.misc
import cv2
import os.path
import tempfile
import time

import benu.utils
import benu.benu

import motmot.FlyMovieFormat.FlyMovieFormat

import roslib

roslib.load_manifest('rospy')
import rospy

roslib.load_manifest('camera_model')
import camera_model

roslib.load_manifest('strawlab_freeflight_experiments')
import analysislib.args
import analysislib.movie
import std_msgs.msg
import geometry_msgs.msg

roslib.load_manifest('flycave')
import autodata.files

roslib.load_manifest('flyvr')
import flyvr.display_client

CALIBRATION = '/home/strawlab/ros-flycave.electric.boost1.46/flycave/calibration/feb2013/colormoviecamcalib.bag'
OSG_FILE = 'posts3.osg'

VR_PANELS = ['virtual_world']

TARGET_OUT_W, TARGET_OUT_H = 1024, 768
MARGIN = 0

def doit(h5_file, fmf_fname, obj_id, tmpdir, outdir, sml):
    h5 = tables.openFile(h5_file, mode='r')
    trajectories = h5.root.trajectories
    dt = 1.0/trajectories.attrs['frames_per_second']
    trajectory_start_times = h5.root.trajectory_start_times

    renderers = {}
    for name in VR_PANELS:
        node = "/ds_%s" % name
        dsc = flyvr.display_client.DisplayServerProxy(display_server_node_name=node,wait=True)
        dsc.set_mode('StimulusOSGFile')
        renderers[name] = flyvr.display_client.RenderFrameSlave(dsc)

    pub_stimulus = rospy.Publisher('stimulus_filename', std_msgs.msg.String, latch=True)
    pub_stimulus.publish(OSG_FILE)

    pose_pub = rospy.Publisher('pose', geometry_msgs.msg.Pose, latch=True)

    # setup camera position
    for name in VR_PANELS:
        # easiest way to get these:
        #   rosservice call /ds_geometry/get_trackball_manipulator_state
        if name=='virtual_world':
            msg = flyvr.msg.TrackballManipulatorState()
            msg.rotation.x = -0.0563853703639
            msg.rotation.y = -0.249313040186
            msg.rotation.z = -0.959619648636
            msg.rotation.w = -0.117447128336
            msg.center.x = -0.00815043784678
            msg.center.y = -0.0655635818839
            msg.center.z = 0.54163891077
            msg.distance = 1.26881285595
        elif name=='geometry':
            msg = flyvr.msg.TrackballManipulatorState()
            msg.rotation.x = 0.122742295197
            msg.rotation.y = 0.198753058426
            msg.rotation.z = 0.873456803025
            msg.rotation.w = 0.427205763051
            msg.center.x = -0.187373220921
            msg.center.y = -0.0946640968323
            msg.center.z = 0.282709181309
            msg.distance = 1.5655520953
        else:
            msg = None

        if msg is not None:
            renderers[name].set_view(msg)

    camera = camera_model.load_camera_from_bagfile( open(CALIBRATION) )

    movie = analysislib.movie.MovieMaker(tmpdir, "%s%s" % (obj_id, sml))

    query = "obj_id == %d" % obj_id
    valid = trajectories.readWhere(query)

    starts = trajectory_start_times.readWhere(query)
    start = starts['first_timestamp_secs'][0] + (starts['first_timestamp_nsecs'][0]*1e-9)

    print "fmf fname", fmf_fname

    fmf = motmot.FlyMovieFormat.FlyMovieFormat.FlyMovie(fmf_fname)
    fmftimes = fmf.get_all_timestamps()

    timestamps = np.arange(
                        start,
                        start+(len(valid)*dt),
                        dt)

    xyz = np.c_[valid['x'],valid['y'],valid['z']]
    pixel = camera.project_3d_to_pixel(xyz)

    print "trajectory ranges from", timestamps[0], "to", timestamps[-1]

    fmftimestamps = fmf.get_all_timestamps()

    print "fmf ranges from", fmftimestamps[0], "to", fmftimestamps[-1]

    num = 0
    t0 = fmf.get_frame_at_or_before_timestamp(timestamps[0])[1]

    if not sml:
        target_out_w = TARGET_OUT_W*2
        target_out_h = TARGET_OUT_H*2
    else:
        target_out_w = TARGET_OUT_W
        target_out_h = TARGET_OUT_H

    #define the size of the output
    device_y0 = MARGIN
    device_y1 = target_out_h-MARGIN
    max_height = device_y1-device_y0

    #define the panels maximum size
    panels = {}
    panels["movie"] = dict(
        width = fmf.width,
        height = fmf.height,
        device_x0 = MARGIN,
        device_x1 = target_out_w//(1+len(VR_PANELS)) - MARGIN//2,
    )
    for n,name in enumerate(VR_PANELS):
        panels[name] = dict(
            width = renderers[name].dsc.width,
            height = renderers[name].dsc.height,
            device_x0 = (1+n)*target_out_w//(1+len(VR_PANELS)) + MARGIN//2,
            device_x1 = (2+n)*target_out_w//(1+len(VR_PANELS)) - MARGIN//2,
        )
    #calculate sizes of the panels that fit
    actual_out_w, actual_out_h = benu.utils.negotiate_panel_size(panels, max_height, MARGIN)

    pbar = analysislib.get_progress_bar(str(obj_id), len(timestamps))

    for n,(t,uv,xyz) in enumerate(zip(timestamps,pixel,xyz)):
        img,ts = fmf.get_frame_at_or_before_timestamp(t)

        if ts > t0:
            t0 = ts

            col,row = uv
            x,y,z = xyz

            #see fmfcat commit for why this is right
            rgb_image = cv2.cvtColor(img[:,1:],cv2.COLOR_BAYER_GR2RGB)
            #and this is wrong?
            #rgb_image = cv2.cvtColor(img,cv2.COLOR_BAYER_BG2RGB)

            imgfname = movie.next_frame()
            canv = benu.benu.Canvas(imgfname,actual_out_w,actual_out_h)
            canv.poly([0,0,actual_out_w,actual_out_w,0],[0,actual_out_h,actual_out_h,0,0], color_rgba=(0,0,0,1))

            #do the movie first
            m = panels["movie"]
            device_rect = (m["device_x0"], device_y0, m["dw"], m["dh"])
            user_rect = (0,0,m["width"], m["height"])
            with canv.set_user_coords(device_rect, user_rect) as _canv:
                _canv.imshow(rgb_image, 0,0, filter='best' )
                _canv.scatter([col], [row], color_rgba=(1,0,0,0.8), radius=6, markeredgewidth=5 )

            #do the VR 
            msg = geometry_msgs.msg.Pose()
            msg.position.x = x
            msg.position.y = y
            msg.position.z = z
            pose_pub.publish(msg)
            time.sleep(0.01) # give message a change to get to display server

            for name in VR_PANELS:
                fn = os.path.basename(imgfname)
                myfname = imgfname.replace(fn,name+fn)
                renderers[name].render_frame(myfname, msg)

                time.sleep(0.01) # disk i/o

                m = panels[name]
                device_rect = (m["device_x0"], device_y0, m["dw"], m["dh"])
                user_rect = (0,0,m["width"], m["height"])
                with canv.set_user_coords(device_rect, user_rect) as _canv:
                    _canv.imshow( scipy.misc.imread(myfname), 0,0, filter='best' )

            pbar.update(n)

            canv.save()

    pbar.finish()

    moviefname = movie.render(outdir)
    movie.cleanup()

    print "wrote", moviefname

if __name__ == "__main__":

    rospy.init_node('render3up')

    obj_ids = [738,3065,1607,1925,1586]
    obj_ids = [738]

    uuid   = 'be130ece9db611e2b8fe6c626d3a008a'

    outdir = '/mnt/strawarchive/John/post-for-lisa-ist/'
    outdir = os.getcwd()

    fm = autodata.files.FileModel()
    fm.select_uuid(uuid)
    h5_fname = fm.get_file_model("simple_flydra.h5").fullpath

    print "h5 fname", h5_fname

    for obj_id in obj_ids:
        tmpdir = tempfile.mkdtemp(str(obj_id), dir="/mnt/ssd/tmp/")

        try:
            fmf_fname = autodata.files.get_fmf_file(uuid, obj_id,"Basler_21266086",raise_exception=True)
            doit(h5_fname, fmf_fname, obj_id, tmpdir, outdir, sml='_sml')
        except autodata.files.NoFile, e:
            print "missing file", e


