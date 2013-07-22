#!/bin/sh

set -e

TDIR=$(mktemp -d)

python scripts/movies/movie-2up-trajectory.py \
    --calibration /home/strawlab/ros-flycave.electric.boost1.46/flycave/calibration/feb2013/colormoviecamcalib.bag \
    --uuid 1d06dfe0a2c711e2b7ca6c626d3a008a --idfilt 1278 --zoom-fly \
    --outdir $TDIR

python scripts/movies/movie-1up.py \
    --calibration /home/strawlab/ros-flycave.electric.boost1.46/flycave/calibration/feb2013/colormoviecamcalib.bag \
    --uuid 1d06dfe0a2c711e2b7ca6c626d3a008a --idfilt 1278 \
    --outdir $TDIR

roslaunch ./scripts/movies/launch/render_trajectory_displayservers.launch &
jobpid=$!
sleep 5

#post osg file
python scripts/movies/movie-osgfile-virtualworld.py \
    --calibration /home/strawlab/ros-flycave.electric.boost1.46/flycave/calibration/feb2013/colormoviecamcalib.bag \
    --uuid be130ece9db611e2b8fe6c626d3a008a \
    --movie-file /mnt/strawscience/movies/Flycave/be130ece9db611e2b8fe6c626d3a008a/Basler_21266086/1607.fmf \
    --idfilt 1607 \
    --stimulus StimulusOSGFile \
    --osgdesc posts3.osg/0,0,0/1,1,1 \
    --zfilt none --rfilt none \
    --outdir $TDIR

##gopro osg file and movie camera
python scripts/movies/movie-osgfile-virtualworld.py \
    --calibration /home/strawlab/ros-flycave.electric.boost1.46/flycave/calibration/feb2013/colormoviecamcalib.bag \
    --uuid 39665d18d81011e292be6c626d3a008a \
    --movie-file /mnt/strawscience/movies/Flycave/39665d18d81011e292be6c626d3a008a/Basler_21266086/16.fmf \
    --idfilt 16 \
    --osgdesc L.osgt/0.0,0.0,0.29/0.1,0.1,0.3 \
    --zfilt none --rfilt none \
    --outdir $TDIR

##gopro osg file and gopro camera
python scripts/movies/movie-osgfile-virtualworld.py \
    --uuid 39665d18d81011e292be6c626d3a008a \
    --movie-file '/mnt/storage/GOPROVIDFRAMES/39665d18d81011e292be6c626d3a008a|60.0|1371559150.51' \
    --idfilt 16 \
    --osgdesc L.osgt/0.0,0.0,0.29/0.1,0.1,0.3 \
    --zfilt none --rfilt none \
    --outdir $TDIR

kill $jobpid
sleep 2

roslaunch ./scripts/movies/launch/render_cylinder_displayservers.launch &
jobpid=$!
sleep 5

##render conflict stimulus with post
python scripts/movies/movie-osgfile-virtualworld.py \
    --uuid e982adaee33311e2b07e6c626d3a008a \
    --idfilt 12617 \
    --stimulus StimulusCylinderAndModel \
    --osgdesc "justpost1.osg|-0.15|0.25|0.0" \
    --plot \
    --zfilt trim --rfilt trim \
    --outdir $TDIR

kill $jobpid

echo "movies in $TDIR"
