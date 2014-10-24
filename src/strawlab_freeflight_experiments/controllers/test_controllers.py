import numpy as np
import scipy.io

from MPC import MPC
from TNF import TNF

from util import Fly

def _test_should_control(m):
    m.reset()

    flies = [Fly(x=124,y=-100,obj_id=3,z=0,vx=0,vy=0,vz=0,heading=0),
             Fly(x=95,y=-0.35,obj_id=4,z=0,vx=0,vy=0,vz=0,heading=0),
             Fly(x=1.6,y=0.05,obj_id=10,z=0,vx=0,vy=0,vz=0,heading=0),
             Fly(x=0.05,y=0.05,obj_id=42,z=0,vx=0,vy=0,vz=0,heading=0)]
    flies_dict = {f.obj_id:f for f in flies}

    assert not m.controller_enabled

    obj_id = m.should_control(flies)
    if obj_id != -1:
        print "chose fly %s (enable:%s)" % (obj_id, m.controller_enabled)
        f = flies_dict[obj_id]
        m.run_ekf(f)
        m.run_control()
        m.run_calculate_input()
        return True

    assert False

def _prep(m):

    m._CT_cur_fly_pos[0] = 0.25
    m._CT_cur_fly_pos[1] = 0.0

    m._CT_ekf_enabled.value = 1
    m._CT_cntrl_enabled.value = 1

    m.run_control()
    m.run_calculate_input()
    m.run_ekf(None, 0.25, 0.0)

def _compare(result, test):
    for k in result:
        print "%s\t\t" % k
        try:
            ok = np.allclose(result[k], test[k])
            print "\tp (shape): %r\n\tm (shape): %r\n\tallclose: %s" % (result[k].shape, test[k].shape, ok)
            if not ok:
                d = (result[k] - test[k])
                print "\n\t-----------DIFFER BY", d.sum()
                print "python:\n",result[k], '\nmatlab:\n', test[k], '\n'
        except AttributeError:
            #scalar
            print "\t",result[k], 'vs', test[k]


def test_tnf():
    m = TNF(ts_d=0.01,ts_ci=0.0125,ts_c=0.025,ts_ekf=0.005)
    _prep(m)

    result = {
        'w_out':m._CT_wout.value,
        'zeta_out':np.fromiter(m._CT_zetaout,np.double),
        'xi_out':np.fromiter(m._CT_xiout,np.double),
        'xest':m._ekf_xest.squeeze(),
        'omegae':m._CT_omegae.value,
        'intstate':m._ctr_intstate.squeeze(),
        'targetPoint':np.array(m.target_point)
    }
    test = scipy.io.loadmat('test/tnf.mat',squeeze_me=True)
    _compare(result,test)

    _test_should_control(m)

def test_mpc():
    m = MPC()
    _prep(m)

    result = {
        'w_out':m._CT_wout.value,
        'theta_out':m._CT_thetaout.value,
        'Jout':m._CT_jout.value,
        'omegae':m._CT_omegae.value,
        'xest':m._ekf_xest.squeeze(),
        'Pminus':m._ekf_pminus,
        'xco':m._proj_cox,
        'u':m._proj_u,
        'x':m._proj_x,
        'targetPoint':np.array(m.target_point)
    }
    test = scipy.io.loadmat('test/mpc.mat',squeeze_me=True)
    _compare(result,test)

    _test_should_control(m)

if __name__ == "__main__":
    print "------------------------------TESTING MPC"
    test_mpc()
    print "------------------------------TESTING TNF"
    test_tnf()

