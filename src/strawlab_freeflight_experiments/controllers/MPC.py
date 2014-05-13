import numpy as np
import os.path
import ctypes as ct
import ctypes.util

lib = ct.cdll.LoadLibrary(os.path.abspath("libmpc.so"))
clib = ct.cdll.LoadLibrary(ctypes.util.find_library("c"))


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

        self._CT_cur_fly_pos = np.zeros(2, dtype=ctypes.c_double)
        
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

        #void contr_subopt_MPC_fly_model2 (double *Jout, double *wout, double *thetaout, int enable, contrp_t *cp, projGrState_t *projGrState, ekfState_t *ekfState, int enableEKF, cInpState_t *cInpState)
        self._ctrfcn = lib.contr_subopt_MPC_fly_model2
        self._ctrfcn.argtypes = [ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double),
                                 ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_void_p,
                                 ct.c_int, ct.c_void_p]

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

    @property
    def ekf_enabled(self):
        return self._CT_ekf_enabled.value
    @property
    def controller_enabled(self):
        return self._CT_cntrl_enabled.value
    @property
    def rotation_rate(self):
        return self._CT_omegae.value

    def run_control(self):
        #void contr_subopt_MPC_fly_model2 (double *Jout, double *wout, double *thetaout, int enable, contrp_t *cp, projGrState_t *projGrState, ekfState_t *ekfState, int enableEKF, cInpState_t *cInpState)
        self._ctrfcn(self._CT_jout, self._CT_wout, self._CT_thetaout,
                     int(self.controller_enabled),
                     self._conp, self._prjs,
                     self._ekfs,
                     int(self.ekf_enabled),
                     self._cins)

    def run_ekf(self, fly):
        self._CT_cur_fly_pos[0] = fly.x
        self._CT_cur_fly_pos[1] = fly.y

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
    import collections

    Fly = collections.namedtuple('Fly', 'x y obj_id')

    m = MPC()
    m.reset()

    flies = [Fly(x=124,y=-100,obj_id=3),
             Fly(x=95,y=-0.35,obj_id=4),
             Fly(x=1.6,y=0.05,obj_id=10),
             Fly(x=0.05,y=0.05,obj_id=42)]
    flies = {f.obj_id:f for f in flies}

    obj_id = m.should_control(flies.values())
    if obj_id != -1:
        f = flies[obj_id]
        m.run_ekf(f)
        m.run_control()
        m.run_calculate_input()

    print m.rotation_rate
#    print lib.init_par_cInpF_decF_ekf_subopt_MPC_model2
