import os
import atexit
import socket
import subprocess

import pymatbridge

def _get_xvfb_lock(process_name):
    global lock_socket
    lock_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        #A unix domain socket in the abstract namesapce, i.e. bound to a name
        #begining with a \0 wil be cleared automarically by the kernel on
        #process destruction.
        lock_socket.bind('\0' + process_name)
        return True
    except socket.error:
        return False

def _start_xvfb(display):
    global xvfb
    xvfb = subprocess.Popen(['/usr/bin/Xvfb', ':%d' % display, '-screen', '0', '1600x1600x24'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    xvfb.poll()
    print "Started Xvfb :%d subprocess (pid %s)" % (display,xvfb.pid)

@atexit.register
def _kill_xvfb():
    global xvfb
    try:
        xvfb.terminate()
    except OSError:
        print "Error killing Xvfb"
    except NameError:
        #not started
        pass

def _get_unique_instance(MAX_INSTANCE=5):
    i = 0
    while i < MAX_INSTANCE:
        #display numbers start at :10 incase I have multiple other
        #xservers running for some reason
        display = i + 10
        if _get_xvfb_lock('xvfb%d' % display):
            _start_xvfb(display)
            return display
        i += 1

@atexit.register
def _kill_mlab():
    global mlab
    try:
        mlab.stop()
    except NameError:
        #not started
        pass
    except Exception, e:
        print "Error killing MATLAB: %s" % e

def get_mlab_instance(visible):
    global mlab
    mlab_opts = dict(matlab='/opt/matlab/R2013a/bin/matlab',
                     capture_stdout=False,
                     log=False)

    if visible:
        mlab = pymatbridge.Matlab(**mlab_opts)
        mlab.start()
    else:
        display = _get_unique_instance()
        mlab_opts['identifier'] = str(display)
        mlab = pymatbridge.Matlab(**mlab_opts)
        mlab_env = os.environ.copy()
        mlab_env['DISPLAY'] = ':%d' % display
        mlab.start(mlab_env)

    return mlab

