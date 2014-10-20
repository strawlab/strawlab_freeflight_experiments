# coding=utf-8
"""Experiment ID, string condition manipulation and genotype condition manipulation from the DCN experiments."""
from itertools import chain
from subprocess import check_call
import os.path as op

from joblib import cpu_count, Parallel, delayed

from flydata.strawlab.experiments import FreeflightExperiment
from flydata.strawlab.metadata import FreeflightExperimentMetadata
from flydata.strawlab.trajectories import FreeflightTrajectory


#################
# UUIDS
#################

REMOVED = (
    '551cdf66ddae11e2a04a6c626d3a008a',  # ato x tshirt-gal80-TNTE (f) NO CONFLICT
    '16e7c2bc0f2511e382596c626d3a008a',  # ato x TNTe_tshirt (f) NO CONFLICT
    '00d53a6e10c811e39ed06c626d3a008a',  # ato x TNTin_tshirt (f) NO CONFLICT
    '0b467d6c0f3011e382596c626d3a008a',  # ato x TNTin_tshirt (f) NO CONFLICT
    'c9aa2afafa2711e384606c626d3a008a',  # ato x TNTE tshirt  DID NOT WORK
    'c4083936f89511e3bd236c626d3a008a',  # VT37804 x TNTin  DID NOT WORK
)


ATO_TNTE = (
    'd4d3a2fa602411e3a3446c626d3a008a',  # atoxTNTE-tshirt (f)
    '6ddf495e5dcd11e385a06c626d3a008a',  # ato x TNTe_tshirt (f)
    'ae8425d4084911e4aafb6c626d3a008a',  # ato x TNTE-tshirt
    'b728144a09dc11e488736c626d3a008a',  # ato x TNTE-tshirt
    'e03d13240ab511e485096c626d3a008a',  # ato x TNTE-tshirt
)


ATO_TNTin = (
    'ef67f4625e9511e398766c626d3a008a',  # atoxTNTin-tshirt (f)
    '44c804fc60ed11e3946c6c626d3a008a',  # atoxTNTin-tshirt(f)
    '09e0b70a61b811e39cdc6c626d3a008a',  # atoxTNTin-tshirt(f)
    '2629a5a8078411e4892a6c626d3a008a',  # ato x TNTin-tshirt
    'c7c23d78091711e4bb2a6c626d3a008a',  # ato xTNTin-tshirt
)


ULTIMATE_TNTE = (
    'f5adba10e8b511e2a28b6c626d3a008a',  # Ultimate TNTE (f)
    'be353e86ea4111e2ae1b6c626d3a008a',  # Ultimate TNTE (f)
    'fe0bdff6f47d11e29bbd6c626d3a008a',  # Ultimate TNTE (f)
)

ULTIMATE_TNTin = (
    '84808eace7e411e2abb66c626d3a008a',  # Ultimate TNTin (f)
)

VT37804_TNTE = (
    '0aba1bb0ebc711e2a2706c626d3a008a',  # VT37804xTNTE (f)
    '7565cafeefc311e293816c626d3a008a',  # VT37804xTNTE (f)
    'c84256845aa811e393ee6c626d3a008a',  # VT37804xTNTE  --> females
    'b006600e5c3511e384a66c626d3a008a',  # VT37804xTNTE  --> females
    'aea9f1e0f7ca11e38cf16c626d3a008a',  # VT37804 x TNTE
)

VT37804_TNTin = (
    '7c8e05c4f2e511e28d866c626d3a008a',  # VT37804xTNTin (f)
    'e20c67fa5b6911e393ee6c626d3a008a',  # VT37804xTNTin --> females
    '6114907c5d0411e3b3a96c626d3a008a',  # VT37804xTNTin (f)
    '735483eef8a111e3a0146c626d3a008a',  # VT37804 x TNTin
    'ad0377f0f95d11e38cd26c626d3a008a',  # VT37804 x TNTin
)

DCN_UUIDs = ATO_TNTE + ATO_TNTin + ULTIMATE_TNTE + ULTIMATE_TNTin + VT37804_TNTE + VT37804_TNTin

DCN_COMPLETED_EXPERIMENTS = ATO_TNTE + ATO_TNTin + VT37804_TNTE + VT37804_TNTin

#################
# Conditions
#################

# A condition with closed-loop rotation and a post in a corner
DCN_CONFLICT_CONDITION = 'checkerboard16.png/infinity.svg/+0.3/-10.0/0.1/0.20/justpost1.osg|-0.15|0.25|0.0'
DCN_ROTATION_CONDITION = 'checkerboard16.png/infinity.svg/+0.3/-10.0/0.1/0.20'

# (x, y) for the post center in arena coordinates
DCN_POST_CENTER = (-0.15, 0.25)

#################
# Workaround for the following problem:
# checkerboard16.png/infinity.svg/+0.3/-10.0/0.1/0.20
# and...
# checkerboard16.png/infinity.svg/+0.3/-10.0/0.1/0.20/
# are the same condition (bar for different implementations at different times and the trailing "/")
#
# Temporary solution: remove trailing /, but keep it for accessing
# Better solution: normalize names in datafiles
#################


def normalize_condition_string(condition):
    """Workaround for trailing bars in condition names.
    These two should be regarded as the same condition:
    >>> cond1 = 'checkerboard16.png/infinity.svg/+0.3/-10.0/0.1/0.20'
    >>> cond2 = 'checkerboard16.png/infinity.svg/+0.3/-10.0/0.1/0.20/'
    >>> cond1 == normalize_condition_string(cond1)
    True
    >>> cond1 == normalize_condition_string(cond2)
    True
    """
    if isinstance(condition, FreeflightTrajectory):
        condition = condition.condition()
    return condition[:-1] if condition.endswith('/') else condition


def unnormalize_condition_name(condition, known_conditions):
    """Workaround for trailing bars in condition names.
    These two should be regarded as the same condition in general,
    but they are not when used as keys to dictionaries:
    >>> cond1 = 'checkerboard16.png/infinity.svg/+0.3/-10.0/0.1/0.20'
    >>> cond2 = 'checkerboard16.png/infinity.svg/+0.3/-10.0/0.1/0.20/'
    >>> conditions = [cond2]
    >>> cond2 == unnormalize_condition_name(cond1, conditions)
    True
    """
    if isinstance(condition, FreeflightTrajectory):
        condition = condition.condition()
    return condition + '/' if condition + '/' in known_conditions else condition

#############
# -End of workaround


def dcn_conflict_select_columns(
    rotation_rate=True,  # speed of the stimulus rotation
    trg_x=True,          # x towards which the rotation stimulus is pushing the fly to
    trg_y=True,          # y towards which the rotation stimulus is pushing the fly to
    trg_z=True,          # z towards which the rotation stimulus is pushing the fly to
    cyl_x=False,
    cyl_y=False,
    cyl_r=False,
    ratio=True,          # [0,1] infinity loop position from the center of the infinity figure
    v_offset_rate=False,
    phase=False,
    model_x=False,
    model_y=False,
    model_z=False,
    model_filename=False,
    condition=False,
    lock_object=False,
    t_sec=False,
    t_nsec=False,
    flydra_data_file=False,
    exp_uuid=False,
    # Response information
    x=True,
    y=True,
    z=True,
    tns=False,
    vx=True,
    vy=True,
    vz=True,
    velocity=True,
    ax=True,
    ay=True,
    az=True,
    theta=True,
    dtheta=True,
    radius=True,
    omega=True,
    rcurve=True,
    framenumber=False
):
    """Returns a list with the names of the parameters that are set to True
    (which correspond to columns names in the combined dataframes)."""
    return [column for column, useful in sorted(locals().items()) if useful]


#################
# Genotypes
#################

def genotype_for_uuid(uuid):
    """
    Returns the genotype string for a given experiment ID.

    Examples:
    >>> print(genotype_for_uuid('551cdf66ddae11e2a04a6c626d3a008a'))
    ato x tshirt-gal80-TNTE (f)
    >>> print(genotype_for_uuid('0aba1bb0ebc711e2a2706c626d3a008a'))
    VT37804xTNTE (f)

    Note that this function requires access to our strawlab-vlan.
    At the moment we assume there is a single genotype tested per experiment.
    """
    # The right place for this is the database.
    # Scrapping the web is just a lame way to go...
    import requests
    text = requests.get(u'http://strawcore:8080/experiment/freeflight_flycave/%s' % str(uuid)).text
    return text.partition('Genotype:</b>')[2].partition('</p>')[0].strip()


def is_female_only(genotype):
    """Were the flies in the experiment exclusively females?

    Examples:
    >>> is_female_only('ato x tshirt-gal80-TNTE (f)')
    True
    >>> is_female_only('ato x tshirt-gal80-TNTE')
    False
    """
    return genotype.lower().endswith('(f)')


def is_tshirt(genotype):
    """t-shirt means that expression of toxins is better delimited to the targeted neurons.
    Usually using the gal80 inhibitor.

    Examples:
    >>> is_tshirt('ato x tshirt-gal80-TNTE (f)')
    True
    >>> is_tshirt('ato x TShirt-gal80-TNTE')
    True
    >>> is_tshirt('ato x TNTE')
    False
    """
    return 'tshirt' in genotype.lower()


def is_ato(genotype):
    """ato genotypes should have impaired only the DCNs.

    Examples:
    >>> is_ato('ato x tshirt-gal80-TNTE (f)')
    True
    >>> is_ato('VT37804xTNTE (f)')
    False
    """
    return 'ato' in genotype.lower()


def is_ultimate(genotype):
    """ultimate genotype is, at the moment, the one where TNT is better delimited to the optic tubercle only.

    Examples:
    >>> is_ultimate('ato x tshirt-gal80-TNTE (f)')
    False
    >>> is_ultimate('Ultimate TNTin (f)')
    True
    """
    return 'ultimate' in genotype.lower()


def is_vt37804(genotype):
    """
    VT37804 is the original line in the experiments, TNT is expressed in both the DCNs and the optic tubercle.

    Examples:
    >>> is_vt37804('ato x tshirt-gal80-TNTE (f)')
    False
    >>> is_vt37804('VT37804xTNTin (f)')
    True
    """
    return 'vt37804' in genotype.lower()


def is_tnte(genotype):
    """Is TNT expressed in the targeted cells?
    That is, are the targeted cells impaired?

    Examples:
    >>> is_tnte('ato x tshirt-gal80-TNTE (f)')
    True
    >>> is_tnte('VT37804xTNTin (f)')
    False
    """
    return 'tnte' in genotype.lower()


def is_dcn(genotype):
    """For the given genotype, are the toxins expressed in the DCN neurons?
    Note that this includes genotypes in which not only DCNs but other areas can be impaired, like VT37804.

    Examples:
    >>> is_dcn('ato x tshirt-gal80-TNTE (f)')
    True
    >>> is_dcn('VT37804xTNTin (f)')
    True
    >>> is_dcn('Ultimate TNTin (f)')
    False
    """
    return is_vt37804(genotype) or is_ato(genotype)


def is_dcn_only(genotype):
    """For the given genotype, are the toxins expressed *only* in the DCN neurons?

    Examples:
    >>> is_dcn_only('ato x tshirt-gal80-TNTE (f)')
    True
    >>> is_dcn_only('VT37804xTNTin (f)')
    False
    >>> is_dcn_only('Ultimate TNTin (f)')
    False
    """
    return is_ato(genotype)


def is_dcn_only_impaired(genotype):
    """Given a genotype, were only the DCNs impaired?

    Examples:
    >>> is_dcn_only_impaired('ato x tshirt-gal80-TNTE (f)')
    True
    >>> is_dcn_only_impaired('ato x tshirt-gal80-TNTin (f)')
    False
    >>> is_dcn_only_impaired('VT37804xTNTE (f)')
    False
    >>> is_dcn_only_impaired('VT37804xTNTin (f)')
    False
    >>> is_dcn_only_impaired('Ultimate TNTE (f)')
    False
    """
    return is_dcn_only(genotype) and is_tnte(genotype)


def is_dcn_impaired(genotype):
    """Given a genotype, were the DCNs impaired?

    Examples:
    >>> is_dcn_impaired('ato x tshirt-gal80-TNTE (f)')
    True
    >>> is_dcn_impaired('ato x tshirt-gal80-TNTin (f)')
    False
    >>> is_dcn_impaired('VT37804xTNTE (f)')
    True
    >>> is_dcn_impaired('VT37804xTNTin (f)')
    False
    >>> is_dcn_impaired('Ultimate TNTE (f)')
    False
    """
    return is_dcn(genotype) and is_tnte(genotype)


def is_optic_tubercle_only(genotype):
    """Expression on the optic-turbecle only?"""
    return is_ultimate(genotype)


def is_optic_tubercle_only_impaired(genotype):
    """Expression on the optic-turbecle only, and expressing TNT?"""
    return is_optic_tubercle_only(genotype) and is_tnte(genotype)


def normalize_genotype_string(genotype):
    """Normalizes the genotype strings so that flies with the same genotype get the same string."""
    if isinstance(genotype, FreeflightExperimentMetadata):
        genotype = genotype.genotype()
    if is_ato(genotype) and is_tnte(genotype):
        return 'ATOxTNTE'
    if is_ato(genotype) and not is_tnte(genotype):
        return 'ATOxTNTin'
    if is_vt37804(genotype) and is_tnte(genotype):
        return 'VT37804xTNTE'
    if is_vt37804(genotype) and not is_tnte(genotype):
        return 'VT37804xTNTin'
    if is_ultimate(genotype) and is_tnte(genotype):
        return 'ULTIMATExTNTE'
    if is_ultimate(genotype) and not is_tnte(genotype):
        return 'ULTIMATExTNTin'
    raise Exception('Unknown genotype for %s' % genotype)


DCN_GENOTYPES = (
    'ATOxTNTE',
    'ATOxTNTin',
    'VT37804xTNTE',
    'VT37804xTNTin',
    # 'ULTIMATExTNTE',
    # 'ULTIMATExTNTin',
)


#################
# Initial data combination and filtering
#################

def recombine_csv_with_hdf5(analysis_script='straw-conflict',
                            uuids=DCN_UUIDs,
                            arenas='flycave',
                            zfilt_min=0.1,
                            zfilt_max=0.9,
                            rfilt_max=0.42,
                            lenfilt=1,
                            outdir=None,
                            run=False,
                            n_jobs=None):
    """Generate commands to combine&analyse scripts, optionally running them."""
    # FIXME: handle overwrite by deleting, if not handled by John's code already
    STRAW_ROOT = '/opt/ros/ros-flycave.electric.boost1.46/strawlab_freeflight_experiments/scripts/'
    SANTI_ROOT = '~/Proyectos/imp/software/strawlab_freeflight_experiments/scripts/'
    KNOWN_SCRIPTS = {
        'straw-conflict': op.join(STRAW_ROOT, 'conflict-analysis.py'),
        'santi-conflict': op.join(SANTI_ROOT, 'conflict-analysis.py')
    }
    other_analysis_script = KNOWN_SCRIPTS.get(analysis_script, None)
    if other_analysis_script is not None:
        analysis_script = other_analysis_script
    if isinstance(uuids, basestring):
        uuids = [uuids]
    if isinstance(arenas, basestring):
        arenas = [arenas] * len(uuids)
    if len(arenas) != len(uuids):
        raise Exception('There should be the same number of arenas as of uuids (%d != %d)' %
                        (len(arenas), len(uuids)))
    commands = []
    for arena, uuid in zip(arenas, uuids):
        outdir_uuid = '' if outdir is None else op.join(outdir, uuid)
        commands.append('%s ' % analysis_script +
                        '--uuid %s ' % uuid +
                        '--arena %s ' % arena +
                        '--zfilt trim '
                        '--zfilt-max %g ' % zfilt_max +
                        '--zfilt-min %g ' % zfilt_min +
                        '--rfilt trim '
                        '--rfilt-max %g ' % rfilt_max +
                        '--lenfilt %g ' % lenfilt +
                        ('' if not outdir_uuid else '--outdir %s ' % outdir_uuid) +
                        '&>~/combine-%s.log' % uuid)

    print 'Commands:\n%s' % '\n'.join(commands)

    if run:
        if n_jobs is None:
            n_jobs = cpu_count()
        print 'Running...'
        Parallel(n_jobs=n_jobs)(delayed(check_call)(cl, shell=True) for cl in commands)


#################
# Data loading and strings normalization
#################

def load_lisa_dcn_experiments(uuids, cache_root_dir=None):
    """Loads FreeflightExperiment objects for the DCN experiments, normalizing the condition and genotype strings.

    Parameters
    ----------
    uuids: string or list of strings
        The UUIDs we want to load

    Returns
    -------
    A list of FreeflightExperiment objects for the uuids
    """
    if isinstance(uuids, basestring):
        uuids = [uuids]
    return [FreeflightExperiment(uuid=uuid,
                                 cache_root_dir=cache_root_dir,
                                 md_transformers={'genotype': normalize_genotype_string},
                                 traj_transformers={'condition': normalize_condition_string})
            for uuid in uuids]


def load_lisa_dcn_trajectories(uuids, cache_root_dir=None):
    experiments = load_lisa_dcn_experiments(uuids, cache_root_dir=cache_root_dir)
    return list(chain(*[exp.trajectories() for exp in experiments]))


def print_analysis_parameters():
    print 'exp-date | analysis-date | analysis-cl'
    for exp in load_lisa_dcn_experiments(DCN_UUIDs):
        print '%s | %s | %s' % (
            exp.md().start_asdatetime().strftime('%d %b %Y %X'),
            exp.sfff().pkl_modification_datetime().strftime('%d %b %Y %X'),
            exp.sfff().analysis_command_line())


if __name__ == '__main__':
    import argh
    parser = argh.ArghParser()
    parser.add_commands([
        recombine_csv_with_hdf5,
        print_analysis_parameters
    ])
    parser.dispatch()