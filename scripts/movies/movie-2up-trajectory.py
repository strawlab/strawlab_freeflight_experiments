import pandas
import tables
import numpy as np
import scipy.misc
import cv2
import os.path
import tempfile
import sh
import time
import shutil
import collections

import benu.benu
import benu.utils

from mpl_toolkits.mplot3d import axes3d

import motmot.FlyMovieFormat.FlyMovieFormat

import roslib

roslib.load_manifest('rosbag')
import rosbag

roslib.load_manifest('camera_model')
import camera_model

roslib.load_manifest('flycave')
import strawlab.constants
import autodata.files

roslib.load_manifest('strawlab_freeflight_experiments')
import analysislib.args
import analysislib.movie

TARGET_OUT_W, TARGET_OUT_H = 1024, 768
MARGIN = 0

ZOOM_REGION_WH = 50
ZOOM_REGION_DISPLAY_WH = 100

def doit(h5_file, fmf_fname, obj_id, tmpdir, outdir, calibration, tfix, show_framenumber, zoom_fly):
    h5 = tables.openFile(h5_file, mode='r')
    trajectories = h5.root.trajectories
    dt = 1.0/trajectories.attrs['frames_per_second']
    trajectory_start_times = h5.root.trajectory_start_times

    camera = camera_model.load_camera_from_bagfile( open(calibration) )

    if not os.path.isfile(fmf_fname):
        raise IOError(fmf_fname)

    movie = analysislib.movie.MovieMaker(
                tmpdir,
                ("%s_2up" % obj_id) + ("_zoom" if zoom_fly else "")
    )

    query = "obj_id == %d" % obj_id
    valid = trajectories.readWhere(query)

    starts = trajectory_start_times.readWhere(query)
    start = starts['first_timestamp_secs'][0] + (starts['first_timestamp_nsecs'][0]*1e-9)

    if tfix:
        print "adjusting times by", tfix
        start += tfix

    print "fmf fname", fmf_fname

    fmf = motmot.FlyMovieFormat.FlyMovieFormat.FlyMovie(fmf_fname)
    fmftimes = fmf.get_all_timestamps()

    timestamps = np.arange(
                        start,
                        start+(len(valid)*dt),
                        dt)

    xyz = np.c_[valid['x'],valid['y'],valid['z']]
    pixel = camera.project_3d_to_pixel(xyz)

    fmftimestamps = fmf.get_all_timestamps()

    print "trajectory ranges from", timestamps[0], "to", timestamps[-1]
    print "fmf ranges from", fmftimestamps[0], "to", fmftimestamps[-1]

    t0 = max(timestamps[0],fmftimestamps[0])
    if t0 > min(timestamps[-1],fmftimestamps[-1]):
        raise IOError("%s (contains no overlapping time period)" % fmf_fname)

    #define the size of the output
    device_y0 = MARGIN
    device_y1 = TARGET_OUT_H-MARGIN
    max_height = device_y1-device_y0

    #define the panels maximum size
    panels = {}
    panels["movie"] = dict(
        width = fmf.width,
        height = fmf.height,
        device_x0 = MARGIN,
        device_x1 = 0.5*TARGET_OUT_W - MARGIN//2,
    )
    panels["plot"] = dict(
        width = 500,
        height = 400,
        device_x0 = 0.5*TARGET_OUT_W + MARGIN//2,
        device_x1 = 1.0*TARGET_OUT_W - MARGIN//2,
    )
    actual_out_w, actual_out_h = benu.utils.negotiate_panel_size(panels, max_height, MARGIN)

    pbar = analysislib.get_progress_bar(str(obj_id), len(timestamps))

    MAXLEN = None
    xhist = collections.deque(maxlen=MAXLEN)
    yhist = collections.deque(maxlen=MAXLEN)
    zhist = collections.deque(maxlen=MAXLEN)

    tfirst = None
    for n,(t,uv,xyz,framenumber) in enumerate(zip(timestamps,pixel,xyz,valid['framenumber'])):

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

            xhist.append(x)
            yhist.append(y)
            zhist.append(z)

            imgfname = movie.next_frame()
            canv = benu.benu.Canvas(imgfname,actual_out_w,actual_out_h)
            canv.poly([0,0,actual_out_w,actual_out_w,0],[0,actual_out_h,actual_out_h,0,0], color_rgba=(0,0,0,1))

            #see fmfcat commit for why this is right
            rgb_image = cv2.cvtColor(img[:,1:],cv2.COLOR_BAYER_GR2RGB)
            #and this is wrong?
            #rgb_image = cv2.cvtColor(img,cv2.COLOR_BAYER_BG2RGB)

            #do the movie first
            m = panels["movie"]
            device_rect = (m["device_x0"], device_y0, m["dw"], m["dh"])
            user_rect = (0,0,m["width"], m["height"])
            with canv.set_user_coords(device_rect, user_rect) as _canv:
                _canv.imshow(rgb_image, 0,0, filter='best' )
                _canv.scatter([col], [row], color_rgba=(1,0,0,0.8), radius=6, markeredgewidth=5 )

            if zoom_fly:
                #do the zoomed in part
                z_w = ZOOM_REGION_WH//2
                z_wd = ZOOM_REGION_DISPLAY_WH
                z_image = rgb_image[row-z_w:row+z_w,col-z_w:col+z_w,:]
                device_rect = (m["device_x0"], device_y0+m["dh"]-z_wd, z_wd, z_wd)
                user_rect = (0,0,z_image.shape[1], z_image.shape[0])
                with canv.set_user_coords(device_rect, user_rect) as _canv:
                    _canv.imshow(z_image, 0,0, filter='nearest')

            m = panels["plot"]
            device_rect = (m["device_x0"], device_y0, m["dw"], m["dh"])
            user_rect = (0,0,m["width"], m["height"])
            with canv.set_user_coords(device_rect, user_rect) as _canv:
                with _canv.get_figure(m["dw"], m["dh"]) as fig:
                        ax = fig.add_subplot(111, projection='3d')

                        AZIMUTH_HOME = 48
                        wiggle = 8 * np.sin(0.03*movie.frame_number)

                        ax.view_init(45, AZIMUTH_HOME + wiggle)

                        ax.plot( xhist, yhist, zhist, 'w-', lw=1.5, alpha=1.0, rasterized=False)
                        ax.plot( [x], [y], [z], 'ro', rasterized=False )

                        ax.set_xlabel('')    
                        ax.set_ylabel('')    
                        ax.set_zlabel('')    

                        for rad in [0.5]:
                            theta = np.linspace(0, 2*np.pi, 100)
                            ax.plot( rad*np.cos(theta), rad*np.sin(theta), np.zeros_like(theta), 'r-',
                                     lw=2, alpha=0.5 )
                            ax.plot( rad*np.cos(theta), rad*np.sin(theta), np.ones_like(theta), 'r-',
                                     lw=2, alpha=0.5 )

                        benu.utils.set_foregroundcolor(ax, 'white')
                        benu.utils.set_backgroundcolor(ax, 'black')

                        ax.w_xaxis.set_pane_color((0.3, 0.3, 0.3, 1.0))
                        ax.w_yaxis.set_pane_color((0.3, 0.3, 0.3, 1.0))
                        ax.w_zaxis.set_pane_color((0.3, 0.3, 0.3, 1.0))
                        for a in (ax.w_xaxis, ax.w_yaxis, ax.w_zaxis):
                            a._axinfo['grid']['color'] = (0.6, 0.6, 0.6, 1.0)

                        #ax.grid(False)
                        for a in (ax.w_xaxis, ax.w_yaxis, ax.w_zaxis):
                            for t in a.get_ticklines():
                                t.set_visible(False)
                        #        t.set_color('green')
                            for t in a.get_ticklabels():
                                t.set_visible(False)
                        #        t.set_color('yellow')
                        #    a.line.set_visible(False)
                        #    a.line.set_color('green')
                        #    a.pane.set_visible(False)
                        #    a.pane.set_color((0.3, 0.3, 0.3, 1.0))

                        fig.patch.set_facecolor('none')
                        #fig.tight_layout() 

                if show_framenumber:
                    _canv.text(str(framenumber),m["dw"]-100,m["dh"]-20, color_rgba=(0.5,0.5,0.5,1.0))
                        
            canv.save()

    pbar.finish()

    moviefname = movie.render(outdir)
    movie.cleanup()

    print "wrote", moviefname


if __name__ == "__main__":
    parser = analysislib.args.get_parser("uuid", "h5-file", "idfilt", "outdir", "basedir")
    parser.add_argument(
        '--fmf-file', type=str, nargs='+',
        help='path to fmf file (if not using --uuid)')
    parser.add_argument(
        '--calibration', type=str, required=True,
        help='path to camera calibration file')
    parser.add_argument(
        '--camera', type=str, default="Basler_21266086",
        help='camera uuid that recorded fmf file')
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
        '--zoom-fly', action='store_true',
        help='render zoomed region around fly')

    args = parser.parse_args()

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

    if args.fmf_file:
        obj_ids = [int(os.path.basename(fmf_file)[:-4]) for fmf_file in args.fmf_file]
        fmf_files = args.fmf_file
    else:
        obj_ids = args.idfilt
        fmf_files = [autodata.files.get_fmf_file(uuid,obj_id,args.camera,raise_exception=False) for obj_id in args.idfilt]

    if not obj_ids:
        parser.error("You must specify --idfilt or --fmf-file")

    print "h5 fname", h5_file

    for obj_id,fmf_fname in zip(obj_ids,fmf_files):
        try:
            doit(h5_file, fmf_fname, obj_id, args.tmpdir, outdir, args.calibration, args.tfix, args.framenumber, args.zoom_fly)
        except IOError, e:
            print "missing file", e


