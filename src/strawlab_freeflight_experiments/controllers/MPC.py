import os.path
import collections

import numpy as np
import numpy.ctypeslib
import ctypes as ct
import ctypes.util

lib = numpy.ctypeslib.load_library("libmpc", os.path.join(os.path.dirname(os.path.abspath(__file__)),'mpc'))
clib = ct.cdll.LoadLibrary(ctypes.util.find_library("c"))

Fly = collections.namedtuple('Fly', 'x y obj_id')

class MPC:
    def __init__(self):

        #these values are shared between python and C, and I need to know their
        #value
        self._CT_ekf_enabled = ct.c_double(0)
        self._CT_cntrl_enabled = ct.c_double(0)
        self._CT_omegae = ct.c_double(0)

        self._CT_jout = ct.c_double(0) #cost functional
        self._CT_wout = ct.c_double(0) #evolution of path param
        self._CT_thetaout = ct.c_double(0) #value of path param as used in control computation

        self._CT_targetpoint = data = (ct.c_double * 2)() #current target point index of path [x,y]
        self._target_x, self._target_y = 0,0

        self._CT_cur_fly_pos = np.zeros(2, dtype=ctypes.c_double) #current fly pos [x,y]
        
#        int decFct (double *xpos, double *ypos, int *id, int arrayLen, int reset,
#                double *enableCntr, double *enableEKF,  
#            contrp_t *cp, ekfp_t *ekfp, decfp_t *decfp, decfState_t *decfState,  
#            projGrState_t *projGrState, ekfState_t *ekfState, double *gammaEstimate) {
        self._decfcn = lib.decFct
        self._decfcn.restype = ct.c_int
        self._decfcn.argtypes = [ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_int), ct.c_int, ct.c_int,
                                 ct.POINTER(ct.c_double), ct.POINTER(ct.c_double),
                                 ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p,
                                 ct.c_void_p, ct.c_void_p, ct.POINTER(ct.c_double)]

        # void ekf_fly_model2 (int enable, ekfState_t *ekfState, double omegae, double *sysOut, ekfp_t *ekfp)
        self._ekffcn = lib.ekf_fly_model2
        self._ekffcn.argtypes = [ct.c_int, ct.c_void_p, ct.c_double, ct.POINTER(ct.c_double), ct.c_void_p]

        #void contr_subopt_MPC_fly_model2 (double *Jout, double *wout, double *thetaout, int enable, contrp_t *cp, projGrState_t *projGrState, ekfState_t *ekfState, int enableEKF, cInpState_t *cInpState, double *targetPoint)
        self._ctrfcn = lib.contr_subopt_MPC_fly_model2
        self._ctrfcn.argtypes = [ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double),
                                 ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_void_p,
                                 ct.c_int, ct.c_void_p,
                                 ct.POINTER(ct.c_double)]

        #void calcInput (cInputp_t *cInputp, cInpState_t *cInpState, projGrState_t *projGrState, contrp_t *cp, double *omegae) {
        self._cinfcn = lib.calcInput
        self._cinfcn.argtype = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.POINTER(ct.c_double)]

        self._ekfp = lib.ekf_new_params()
        self._ekfs = lib.ekf_new_state()

        self._decp = lib.decfct_new_params()
        self._decs = lib.decfct_new_state()

        self._conp = lib.contr_new_params()
        self._prjs = lib.contr_new_state()

        self._cinp = lib.calcinput_new_params()
        self._cins = lib.calcinput_new_state()

        lib.init_par_cInpF_decF_ekf_subopt_MPC_model2(self._conp, self._ekfp, self._decp, self._cinp)

        #allocate memory for internal controller variables:
        lib.allocate_memory_controller(self._prjs, self._conp)

        #initialize EKF: 
        lib.initEKF (self._ekfs, self._ekfp)

        #initialize controller
        lib.initProjGradMethod (self._prjs, self._conp)

        #initialize function calculating the input to the system
        lib.initCalcInput (self._cinp, self._cins)

        #initialize decision function
        lib.initDecFct (self._decp, self._decs)

        #create some numpy arrays to shadow internal state
        n = ct.c_int(0)
        m = ct.c_int(0)

        lib.get_proj_state.argtypes = [ct.c_void_p, ct.c_void_p, ct.POINTER(ct.c_int), ct.POINTER(ct.c_int)]
        lib.get_proj_state.restype = ct.POINTER(ct.c_double)
        x = lib.get_proj_state(self._conp, self._prjs, m, n)
        #trick numpy into making a zero-copy view of a fortran (colum major) 
        #hunk of matlab owned memory. The reshape dance is needed because as_array
        #does not support a strides argumemt
        self._proj_x = numpy.ctypeslib.as_array(x, shape=(m.value*n.value, 1))
        self._proj_x = np.reshape(self._proj_x, (m.value,n.value), order='F')

        lib.get_proj_costate.argtypes = [ct.c_void_p, ct.c_void_p, ct.POINTER(ct.c_int), ct.POINTER(ct.c_int)]
        lib.get_proj_costate.restype = ct.POINTER(ct.c_double)
        x = lib.get_proj_costate(self._conp, self._prjs, m, n)
        self._proj_cox = numpy.ctypeslib.as_array(x, shape=(m.value*n.value, 1))
        self._proj_cox = np.reshape(self._proj_cox, (m.value,n.value), order='F')

        lib.get_proj_input.argtypes = [ct.c_void_p, ct.c_void_p, ct.POINTER(ct.c_int), ct.POINTER(ct.c_int)]
        lib.get_proj_input.restype = ct.POINTER(ct.c_double)
        x = lib.get_proj_input(self._conp, self._prjs, m, n)
        self._proj_u = numpy.ctypeslib.as_array(x, shape=(m.value*n.value, 1))
        self._proj_u = np.reshape(self._proj_u, (m.value,n.value), order='F')

        lib.ekf_get_state_estimate.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int)]
        lib.ekf_get_state_estimate.restype = ct.POINTER(ct.c_double)
        x = lib.ekf_get_state_estimate(self._ekfs, n)
        self._ekf_xest = numpy.ctypeslib.as_array(x, shape=(n.value,1))

        lib.ekf_get_state_covariance.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int), ct.POINTER(ct.c_int)]
        lib.ekf_get_state_covariance.restype = ct.POINTER(ct.c_double)
        x = lib.ekf_get_state_covariance(self._ekfs, m, n)
        self._ekf_pminus = numpy.ctypeslib.as_array(x, shape=(m.value*n.value, 1))
        self._ekf_pminus = np.reshape(self._ekf_pminus, (m.value,n.value), order='F')

        #get a copy of the path
        num = ct.c_int(0)
        lib.get_path.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int)]
        lib.get_path.restype = ct.POINTER(ct.c_double)
        path = lib.get_path(self._conp, num)
        #reshape into num rows and 2 columns (x,y)
        self._path = numpy.ctypeslib.as_array(path, shape=(num.value, 2))

    @property
    def path(self):
        return self._path
    @property
    def ekf_enabled(self):
        return self._CT_ekf_enabled.value
    @property
    def controller_enabled(self):
        return self._CT_cntrl_enabled.value
    @property
    def rotation_rate(self):
        return self._CT_omegae.value
    @property
    def path_progress(self):
        return self._CT_thetaout.value
    @property
    def target_point(self):
        return self._target_x, self._target_y

    def run_control(self):
        #void contr_subopt_MPC_fly_model2 (double *Jout, double *wout, double *thetaout, int enable, contrp_t *cp, projGrState_t *projGrState, ekfState_t *ekfState, int enableEKF, cInpState_t *cInpState)
        self._ctrfcn(self._CT_jout, self._CT_wout, self._CT_thetaout,
                     int(self.controller_enabled),
                     self._conp, self._prjs,
                     self._ekfs,
                     int(self.ekf_enabled),
                     self._cins,
                     self._CT_targetpoint)
        self._target_x, self._target_y = self._CT_targetpoint

    def run_ekf(self, fly, x=None, y=None):
        if x is None:
            x = fly.x
        if y is None:
            y = fly.y

        self._CT_cur_fly_pos[0] = x
        self._CT_cur_fly_pos[1] = y

        self._ekffcn(int(self.ekf_enabled), #grr
                     self._ekfs,
                     self._CT_omegae,
                     self._CT_cur_fly_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                     self._ekfp)

    def run_calculate_input(self):
        self._cinfcn(self._cinp,
                     self._cins,
                     self._prjs,
                     self._conp,
                     ct.byref(self._CT_omegae))

    def should_control(self, flies, x=None, y=None, ids=None):
        if x is None:
            x = [f.x for f in flies]
            y = [f.y for f in flies]
            ids = [f.obj_id for f in flies]

        x = np.array(x, dtype=ctypes.c_double)
        y = np.array(x, dtype=ctypes.c_double)
        ids = np.array(ids, dtype=ctypes.c_int)
        g = np.zeros_like(x, dtype=ctypes.c_double)
        n = len(x)

        fid = self._decfcn(
                     x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                     y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                     ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                     n,
                     0,
                     ct.byref(self._CT_cntrl_enabled), ct.byref(self._CT_ekf_enabled),
                     self._conp, self._ekfp, self._decp, self._decs,
                     self._prjs, self._ekfs,
                     g.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

        return fid

    def reset(self):
        self._decfcn(ctypes.POINTER(ctypes.c_double)(), #null pointers, ignored if reset=1
                     ctypes.POINTER(ctypes.c_double)(),
                     ctypes.POINTER(ctypes.c_int)(),
                     0, #len=0
                     1, #reset=1
                     ct.byref(self._CT_cntrl_enabled), ct.byref(self._CT_ekf_enabled),
                     self._conp, self._ekfp, self._decp, self._decs,
                     self._prjs, self._ekfs,
                     ctypes.POINTER(ctypes.c_double)())


if __name__ == "__main__":

    m = MPC()
    m.reset()

    flies = [Fly(x=124,y=-100,obj_id=3),
             Fly(x=95,y=-0.35,obj_id=4),
             Fly(x=1.6,y=0.05,obj_id=10),
             Fly(x=0.05,y=0.05,obj_id=42)]
    flies = {f.obj_id:f for f in flies}

    print "enabled", m.controller_enabled

    obj_id = m.should_control(flies.values())
    if obj_id != -1:
        print "enabled", m.controller_enabled
        f = flies[obj_id]
        m.run_ekf(f)
        m.run_control()
        m.run_calculate_input()

    print m.target_point

    print m.rotation_rate

    m.reset()
    print "enabled", m.controller_enabled

#    print lib.init_par_cInpF_decF_ekf_subopt_MPC_model2
