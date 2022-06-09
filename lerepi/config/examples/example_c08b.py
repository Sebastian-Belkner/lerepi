import numpy as np
import healpy as hp

from lerepi.core.metamodel.dlensalot import *


dlensalot_model = DLENSALOT_Model(
    job = DLENSALOT_Job(
        build_OBD = True,
        QE_lensrec = False,
        MAP_lensrec = True,
        Btemplate_per_iteration = False,
        map_delensing = False,
        inspect_result = False
    ),
    data = DLENSALOT_Data(
        TEMP_suffix = 'example9',
        OBD_type = 'OBD',
        fg = '00',
        sims = 'cmbs4/08b/caterinaILC_May12',
        zbounds =  ('nmr_relative', 10),
        zbounds_len = ('extend', 5.),   
        nside = 2048,
        BEAM = 2.3,
        lmax_transf = 4000,
        transf = hp.gauss_beam,
        pbounds = [1.97, 5.71],
        tpl = 'template_dense',
    ),
    iteration = DLENSALOT_Iteration(
        K = 'p_p',
        V = '', 
        ITMAX = 12,
        IMIN = 0,
        IMAX = 99,
        nsims_mf = 100,
        OMP_NUM_THREADS = 32,
        Lmin = 4, 
        CG_TOL = 1e-3,
        TOL = 3,
        soltn_cond = lambda it: True,
        lmax_filt = 4096,
        lmin_tlm = 30,
        lmin_elm = 30,
        lmin_blm = 30, #Supress all modes below this value, hacky version of OBD
        lmax_qlm = 4096,
        mmax_qlm = 4096,
        lmax_unl = 4000,
        mmax_unl = 4000,
        lmax_ivf = 3000,
        mmax_ivf = 3000,
        lmin_ivf = 10,
        mmin_ivf = 10,
        LENSRES = 1.7, # Deflection operations will be performed at this resolution
        QE_LENSING_CL_ANALYSIS = False, # Change the following block only if a full, Planck-like QE lensing power spectrum analysis is desired
        STANDARD_TRANSFERFUNCTION = True, # Change the following block only if exotic transferfunctions are desired
        FILTER = 'cinv_sepTP', # Change the following block only if other than cinv_t, cinv_p, ivfs filters are desired
        FILTER_QE = 'sepTP', # Change the following block only if other than sepTP for QE is desired
        ITERATOR = 'constmf' # Choose your iterator. Either pertmf or const_mf
    ),
    geometry = DLENSALOT_Geometry(
        lmax_unl = 4000,
        zbounds = ('nmr_relative', np.inf),
        zbounds_len = ('extend', 5.),
        pbounds = [1.97, 5.71],
        nside = 2048,
        lenjob_geometry = 'thin_gauss',
        lenjob_pbgeometry = 'pbdGeometry',
        ninvjob_geometry = 'healpix_geometry',
        ninvjob_qe_geometry = 'healpix_geometry_qe'
    ),
    chain_descriptor = DLENSALOT_Chaindescriptor(
        p0 = 0,
        p1 = ["diag_cl"],
        p2 = None,
        p3 = 2048,
        p4 = np.inf,
        p5 = None,
        p6 = 'tr_cg',
        p7 = 'cache_mem'
    ),
    stepper = DLENSALOT_Stepper(
        typ = 'harmonicbump',
        lmax_qlm = 4000,
        mmax_qlm = 4000,
        xa = 400,
        xb = 1500
    ),
    map_delensing = DLENSALOT_Mapdelensing(
        edges = 'cmbs4',
        IMIN = 0,
        IMAX = 99,
        ITMAX = 10,
        fg = '00',
        base_mask = 'cmbs4/08b/caterinaILC_May12', # This mask is used to rotate ILC maps
        nlevels = [2., 5.],
        nside = 2048,
        lmax_cl = 2048,
        beam = 2.3,
        lmax_transf = 4000,
        transf = 'gauss',
        Cl_fid = 'ffp10'
    ),
    obd = DLENSALOT_OBD(
        BMARG_LIBDIR = '/global/project/projectdirs/cmbs4/awg/lowellbb/reanalysis/mapphi_intermediate/s08b/'
        BMARG_LIBDIR_buildpath = '/global/cscratch1/sd/sebibel/cmbs4/OBD_matrix/'
        BMARG_LCUT = 200.,
        BMARG_RESCALE = (0.42 / 0.350500) ** 2,
        CENTRALNLEV_UKAMIN = 0.42,
        nlev_t = 0.42/np.sqrt(2),
        nlev_p = 0.42,
        nlev_dep = 10000.,
        inf = 1e4,
        ratio = np.inf,
        mask = ('nlev', 100),
        noisemodel_rhits = '/global/project/projectdirs/cmbs4/awg/lowellbb/reanalysis/mapphi_intermediate/s08b/masks/08b_rhits_positive_nonan.fits', #If OBD used, this must be the exact same map with which tniti was build
        noisemodel_norm = 1.0, #divide noisemodel by this value # TODO not sure if needed
    )
)