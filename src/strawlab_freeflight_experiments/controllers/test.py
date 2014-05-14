import numpy as np
import scipy.io

from MPC import MPC

m = MPC()

m._CT_cur_fly_pos[0] = 0.25
m._CT_cur_fly_pos[1] = 0.0

m._CT_ekf_enabled.value = 1
m._CT_cntrl_enabled.value = 1

m.run_control()
m.run_calculate_input()
m.run_ekf(None, 0.25, 0.0)

result = {
    'w_out':m._CT_wout.value,
    'theta_out':m._CT_thetaout.value,
    'Jout':m._CT_jout.value,
    'omegae':m._CT_omegae.value,
    'xest':m._ekf_xest,
    'Pminus':m._ekf_pminus,
    'xco':m._proj_cox,
    'u':m._proj_u,
    'x':m._proj_x
}

test = scipy.io.loadmat('data.mat')

for k in result:
    print k,
    try:
        ok = np.allclose(result[k], test[k])
        print result[k].shape, test[k].shape, ok
        if not ok:
            d = (result[k] - test[k])
            print "DIFFER BY", d.sum()
            print result[k], '\n', test[k], '\n', d
#        diff = (result[k] - test[k])
#        print diff
#        print "(diff = %f)" % diff.sum()
    except AttributeError:
        print result[k], 'vs', test[k]

print np.allclose(result['x'], test['x'])
