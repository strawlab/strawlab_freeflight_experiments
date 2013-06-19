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
import strawlab.constants

roslib.load_manifest('flyvr')
import flyvr.display_client

VR_PANELS = ['virtual_world']

TARGET_OUT_W, TARGET_OUT_H = 1024, 768
MARGIN = 0

def doit(h5_file, fmf_fname, obj_id, tmpdir, outdir, calibration, tfix, framenumber, sml, osg_file_desc):
    h5 = tables.openFile(h5_file, mode='r')
    trajectories = h5.root.trajectories
    dt = 1.0/trajectories.attrs['frames_per_second']
    trajectory_start_times = h5.root.trajectory_start_times

    renderers = {}
    osgslaves = {}
    for name in VR_PANELS:
        node = "/ds_%s" % name
        dsc = flyvr.display_client.DisplayServerProxy(display_server_node_name=node,wait=True)

        renderers[name] = flyvr.display_client.RenderFrameSlave(dsc)
        osgslaves[name] = flyvr.display_client.OSGFileStimulusSlave(dsc, osg_file_desc)

    # setup camera position
    for name in VR_PANELS:
        # easiest way to get these:
        #   rosservice call /ds_geometry/get_trackball_manipulator_state
        msg = flyvr.msg.TrackballManipulatorState()
        if name=='virtual_world':
            #used for the post movies
            if 0:
                msg.rotation.x = -0.0563853703639
                msg.rotation.y = -0.249313040186
                msg.rotation.z = -0.959619648636
                msg.rotation.w = -0.117447128336
                msg.center.x = -0.00815043784678
                msg.center.y = -0.0655635818839
                msg.center.z = 0.54163891077
                msg.distance = 1.26881285595
            #used for the colored l box, more looking down
            else:
                msg.rotation.x = -0.0530832760665
                msg.rotation.y = -0.0785547480223
                msg.rotation.z = -0.986425433667
                msg.rotation.w = -0.134075281764
                msg.center.x = 0.0064534349367
                msg.center.y = 0.0254454407841
                msg.center.z = 0.522875547409
                msg.distance = 1.00728635582

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

    if calibration:
        camera = camera_model.load_camera_from_bagfile( open(calibration) )
    else:
        camera = None

    movie = analysislib.movie.MovieMaker(
                                tmpdir,
                                "%s%s_%s%s" % (obj_id,
                                               sml,
                                               "_".join(VR_PANELS),
                                               "" if fmf_fname.endswith(".fmf") else "_gopro")
    )

    query = "obj_id == %d" % obj_id
    valid = trajectories.readWhere(query)

    starts = trajectory_start_times.readWhere(query)
    start = starts['first_timestamp_secs'][0] + (starts['first_timestamp_nsecs'][0]*1e-9)

    timestamps = np.arange(
                        start,
                        start+(len(valid)*dt),
                        dt)

    if fmf_fname.endswith(".fmf"):
        print "fmf fname", fmf_fname
        fmf = motmot.FlyMovieFormat.FlyMovieFormat.FlyMovie(fmf_fname)
        movie_is_rgb = False #fixe
    else:
        path,fps,tstart = fmf_fname.split("|")
        print "move from image frames in %s at %s fps (ts=%s)" % (path,fps,tstart)
        fmf = analysislib.movie.ImageDirMovie(path,float(fps),float(tstart))
        movie_is_rgb = fmf.is_rgb


    xyz = np.c_[valid['x'],valid['y'],valid['z']]
    if camera:
        pixel = camera.project_3d_to_pixel(xyz)
    else:
        pixel = np.ones((xyz.shape[0],2),dtype=xyz.dtype)
        pixel.fill(np.nan)

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

    tfirst = None
    for n,(t,uv,xyz) in enumerate(zip(timestamps,pixel,xyz)):

        pbar.update(n)

        try:
            img,ts = fmf.get_frame_at_or_before_timestamp(t)
        except ValueError:
            continue

        if ts > t0:
            t0 = ts
            tfirst = ts if tfirst is None else tfirst

            col,row = uv
            x,y,z = xyz

            if movie_is_rgb:
                rgb_image = img
            else:
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
                if (not np.isnan(col)) and (not np.isnan(row)):
                    _canv.scatter([col], [row], color_rgba=(1,0,0,0.8), radius=6, markeredgewidth=5 )

            #do the VR 
            for name in VR_PANELS:
                renderers[name].set_pose(x=x,y=y,z=z)

                fn = os.path.basename(imgfname)
                myfname = imgfname.replace(fn,name+fn)
                renderers[name].render_frame(myfname, msg)

                time.sleep(0.01) # disk i/o

                m = panels[name]
                device_rect = (m["device_x0"], device_y0, m["dw"], m["dh"])
                user_rect = (0,0,m["width"], m["height"])
                with canv.set_user_coords(device_rect, user_rect) as _canv:
                    _canv.imshow( scipy.misc.imread(myfname), 0,0, filter='best' )

            canv.save()

    pbar.finish()

    moviefname = movie.render(outdir)
    movie.cleanup()

    print "wrote", moviefname

if __name__ == "__main__":
    rospy.init_node('osgrender')

    parser = analysislib.args.get_parser("uuid", "h5-file", "idfilt", "outdir", "basedir")
    parser.add_argument(
        '--movie-file', type=str, required=True,
        help='path to movie file (fmf or mp4)')
    parser.add_argument(
        '--calibration', type=str, required=False,
        help='path to camera calibration file')
    parser.add_argument(
        '--tfix', type=float, default=0.0,
        help='time offset to fixup movie')
    parser.add_argument(
        '--tmpdir', type=str, default='/tmp/',
        help='path to temporary directory')
    parser.add_argument(
        '--no-framenumber', action='store_false', dest="framenumber", default="true",
        help='dont render the framenumber on the video')
    parser.add_argument(
        '--osgdesc', type=str, default='posts3.osg',
        help='osg file descriptor string')

    argv = rospy.myargv()
    args = parser.parse_args(argv[1:])

    if (not args.h5_file) and (not args.uuid):
        parser.error("Specify a UUID or a H5 file")

    if args.uuid is not None:
        if len(args.uuid) > 1:
            parser.error("Only one uuid supported for making movies")

        uuid = args.uuid[0]

        fm = autodata.files.FileModel(basedir=args.basedir)
        fm.select_uuid(uuid)
        h5_file = fm.get_file_model("simple_flydra.h5").fullpath
    else:
        uuid = ''

        h5_file = args.h5_file

    outdir = args.outdir if args.outdir is not None else strawlab.constants.get_move_dir(uuid)

    try:
        assert len(args.idfilt) == 1
        obj_id = args.idfilt[0]
    except:
        obj_id = None

    if obj_id is None:
        parser.error("You must specify --idfilt with a single obj_id")

    doit(h5_file, args.movie_file, obj_id, args.tmpdir, outdir, args.calibration, args.tfix, args.framenumber,
            '_sml',
            args.osgdesc)

##examples
# LISA
#    obj_ids = [738,3065,1607,1925,1586]
#    obj_ids = [738]
#    uuid   = 'be130ece9db611e2b8fe6c626d3a008a'
#    outdir = '/mnt/strawarchive/John/post-for-lisa-ist/'
#    outdir = os.getcwd()
#python movie-osgfile-virtualworld.py --uuid be130ece9db611e2b8fe6c626d3a008a --movie-file /mnt/strawscience/movies/Flycave/be130ece9db611e2b8fe6c626d3a008a/Basler_21266086/738.fmf --calibration /home/strawlab/ros-flycave.electric.boost1.46/flycave/calibration/feb2013/colormoviecamcalib.bag --idfilt 738 --outdir /tmp/

##Colored BOX GoPro
#python movie-osgfile-virtualworld.py --uuid 39665d18d81011e292be6c626d3a008a --movie-file '/mnt/ssd/tmp/GOPRO/|60.0|1371559150.51' --idfilt 16 --outdir . --tmpdir /mnt/ssd/tmp/ --osgdesc L.osgt/0.0,0.0,0.29/0.1,0.1,0.3

##Colored BOX Movie cam
#python movie-osgfile-virtualworld.py --uuid 39665d18d81011e292be6c626d3a008a --movie-file /mnt/strawscience/movies/Flycave/39665d18d81011e292be6c626d3a008a/Basler_21266086/16.fmf --calibration /home/strawlab/ros-flycave.electric.boost1.46/flycave/calibration/feb2013/colormoviecamcalib.bag --idfilt 16 --outdir . --tmpdir /mnt/ssd/tmp/ --osgdesc L.osgt/0.0,0.0,0.29/0.1,0.1,0.3


