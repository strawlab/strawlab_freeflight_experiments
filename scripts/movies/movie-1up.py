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

def doit(h5_file, fmf_fname, obj_id, tmpdir, outdir, calibration, tfix):
    h5 = tables.openFile(h5_file, mode='r')
    trajectories = h5.root.trajectories
    dt = 1.0/trajectories.attrs['frames_per_second']
    trajectory_start_times = h5.root.trajectory_start_times

    camera = camera_model.load_camera_from_bagfile( open(calibration) )

    if not os.path.isfile(fmf_fname):
        raise IOError(fmf_fname)

    movie = analysislib.movie.MovieMaker(tmpdir, str(obj_id))

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

    try:
        t0 = fmf.get_frame_at_or_before_timestamp(timestamps[0])[1]
    except ValueError, e:
        raise IOError(fmf_fname)

    pbar = analysislib.get_progress_bar(str(obj_id), len(timestamps))

    for n,(t,uv,xyz) in enumerate(zip(timestamps,pixel,xyz)):
        img,ts = fmf.get_frame_at_or_before_timestamp(t)

        if ts > t0:
            t0 = ts

            col,row = uv
            x,y,z = xyz

            imgfname = movie.next_frame()

            #see fmfcat commit for why this is right
            rgb_image = cv2.cvtColor(img[:,1:],cv2.COLOR_BAYER_GR2RGB)
            #and this is wrong?
            #rgb_image = cv2.cvtColor(img,cv2.COLOR_BAYER_BG2RGB)

            #color is BGR
            cv2.circle(rgb_image, tuple(map(int,uv)), 10, (0, 0, 255), 3)
            cv2.imwrite(imgfname,rgb_image)

        pbar.update(n)

    pbar.finish()

    moviefname = movie.render(os.path.join(outdir,uuid))
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
        '--tfix', type=float, default=0.0,
        help='path to camera calibration file')
    parser.add_argument(
        '--tmpdir', type=str, default='/tmp/',
        help='path to temporary directory')

    args = parser.parse_args()
    outdir = args.outdir if args.outdir is not None else strawlab.constants.MOVIE_DIR

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

    if args.fmf_file:
        obj_ids = [int(os.path.basename(fmf_file)[:-4]) for fmf_file in args.fmf_file]
        fmf_files = args.fmf_file
    else:
        obj_ids = args.idfilt
        fmf_files = [autodata.files.get_fmf_file(uuid,obj_id,raise_exception=False) for obj_id in args.idfilt]

    if not obj_ids:
        parser.error("You must specify --idfilt or --fmf-file")

    print "h5 fname", h5_file

    for obj_id,fmf_fname in zip(obj_ids,fmf_files):
        try:
            doit(h5_file, fmf_fname, obj_id, args.tmpdir, outdir, args.calibration, args.tfix)
        except IOError, e:
            print "missing file", e


