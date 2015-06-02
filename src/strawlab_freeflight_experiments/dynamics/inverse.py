import os.path
import subprocess
import tempfile
import shutil

import scipy.io

MCR_DESC = 'linux 64bit MCR_DESC v8.1 runtime'
MCR_DIR = '/opt/matlab/MCR/v81/'
INV_DYN_PROG_PATH = '/opt/matlab/Inverse_dyn'


def call_inverse_dynamics_program(binpath, mcrpath, matfile, ts, window_size, full_model, reduced_model, save_models, plot_models, quiet=False):
    if not os.path.isfile(binpath):
        raise ValueError('Missing program')
    if not os.path.isdir(mcrpath):
        raise ValueError('Missing %s' % MCR_DESC)

    binpath = os.path.abspath(binpath)
    mcrpath = os.path.abspath(mcrpath)
    matfile = os.path.abspath(matfile)

    #generate LD_LIBRARY_PATH
    ldp = [os.path.join(mcrpath,p) for p in ('runtime/glnxa64',
                                             'bin/glnxa64',
                                             'sys/os/glnxa64',
                                             'sys/java/jre/glnxa64/jre/lib/amd64/native_threads',
                                             'sys/java/jre/glnxa64/jre/lib/amd64/server',
                                             'sys/java/jre/glnxa64/jre/lib/amd64/client',
                                             'sys/java/jre/glnxa64/jre/lib/amd64')]
    old_ldp = os.environ.get('LD_LIBRARY_PATH','.')
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = old_ldp + ':' + ':'.join(ldp)

    env['XAPPLRESDIR'] = os.path.join(mcrpath,'X11/app-defaults')

    if quiet:
        try:
            stdout = subprocess.DEVNULL #python > 3.3
        except AttributeError:
            stdout = open(os.devnull, 'w')
    else:
        stdout = None

    args = [binpath, matfile, str(ts), str(window_size), str(int(full_model)), str(int(reduced_model)), str(int(save_models)), str(int(plot_models))]
    subprocess.check_call(args, env=env, stdout=stdout)

    out = []
    if full_model and save_models:
        out.append(matfile[:-4] + '_dynamics_full.mat')
    if reduced_model and save_models:
        out.append(matfile[:-4] + '_dynamics_reduced.mat')
    return out

def compute_inverse_dynamics_matlab(df, dt, window_size, full_model):
    tdir = tempfile.mkdtemp()

    #save the dataframe to a mat file
    traj_path = os.path.join(tdir,'trajectory.mat')
    dict_df = df.to_dict('list')
    dict_df['index'] = df.index.values
    scipy.io.savemat(traj_path, dict_df, oned_as='column')

    dest_files = call_inverse_dynamics_program(
                                  INV_DYN_PROG_PATH, MCR_DIR,
                                  traj_path, dt,
                                  window_size=window_size,
                                  full_model=full_model,
                                  reduced_model=not full_model,
                                  save_models=True, plot_models=False, quiet=True)

    #we write either the full or the reduced, not both
    array = scipy.io.loadmat(dest_files[0], squeeze_me=True)

    for s in ('Fx','Fy','Fz','T_phi','T_theta','T_eta'):
        df['invdyn_'+s] = array[s]

    shutil.rmtree(tdir)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
                        description='Wrapper for Inverse_dyn MATLAB binary.',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('matfile',nargs=1,help='path to exported trajectory mat file (see export-dataframes.py)')
    parser.add_argument('--binpath', 
                        default=INV_DYN_PROG_PATH,
                        help='path to binary')
    parser.add_argument('--mcrpath',
                        default=MCR_DIR,
                        help='path to %s' % MCR_DESC)
    parser.add_argument('--ts', type=float,
                        default=0.01,
                        help='sample time')
    parser.add_argument('--window_size', type=int,
                        default=25,
                        help='smoothing window size')
    parser.add_argument('--full-model', type=int, choices=(0,1),
                        default=1,
                        help='compute full model')
    parser.add_argument('--reduced-model', type=int, choices=(0,1),
                        default=1,
                        help='compute reduced model')
    parser.add_argument('--save-models', type=int, choices=(0,1),
                        default=1,
                        help='save model mat files')
    parser.add_argument('--plot-models', action='store_true',
                        default=False)

    args = parser.parse_args()

    out = call_inverse_dynamics_program(args.binpath,args.mcrpath,
                                  args.matfile[0],args.ts,
                                  args.window_size,
                                  args.full_model,args.reduced_model,
                                  args.save_models,args.plot_models)


