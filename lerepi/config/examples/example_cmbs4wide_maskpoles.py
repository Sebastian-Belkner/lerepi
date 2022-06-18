import os
from os.path import join as opj
import numpy as np
import healpy as hp

from lerepi.core.metamodel.dlensalot_v2 import *

from plancklens.sims import phas, planck2018_sims


dlensalot_model = DLENSALOT_Model(
    job = DLENSALOT_Job(
        build_OBD = False,
        QE_lensrec = False,
        MAP_lensrec = True,
        Btemplate_per_iteration = True,
        map_delensing = True,
        inspect_result = False
    ),
    analysis = DLENSALOT_Analysis(
        TEMP_suffix = '',
        K = 'p_p',
        V = 'mf07',
        ITMAX = 12,
        IMIN = 0,
        IMAX = 99,
        nsims_mf = 100,
        OMP_NUM_THREADS = 16,
        LENSRES = 1.7, # Deflection operations will be performed at this resolution
    ),
    data = DLENSALOT_Data(
        sims = ('plancklens', 'sims', 'cmb_maps_nlev'),
        sims_settings = {
            'sims_cmb_len': planck2018_sims.cmb_len_ffp10(),
            'cl_transf': hp.gauss_beam(1.0 / 180 / 60 * np.pi, lmax=4096),
            'nlev_t': 0.5/np.sqrt(2),
            'nlev_p': 0.5,
            'nside': 2048,
            'lib_dir': None,
            'pix_lib_phas': phas.pix_lib_phas(opj(os.environ['HOME'], 'pixphas_nside2048'), 3, (hp.nside2npix(2048),))
        },
        lmax_unl = 4000,
        zbounds =  ('nmr_relative', np.inf),
        zbounds_len = ('extend', 5.),   
        pbounds = [1.97, 5.71],
        STANDARD_TRANSFERFUNCTION = True, # Change the following block only if exotic transferfunctions are desired
        Lmin = 4, 
        lmax_filt = 4000,
        lmax_unl = 4000,
        mmax_unl = 4000,
        lmax_ivf = 3000,
        mmax_ivf = 3000,
        lmin_ivf = 10,
        mmin_ivf = 10
    ),
    noisemodel = DLENSALOT_Noisemodel(
        typ = 'OBD',
        BMARG_LIBDIR = '/global/project/projectdirs/cmbs4/awg/lowellbb/reanalysis/mapphi_intermediate/s08b/',
        BMARG_LCUT = 200,
        BMARG_RESCALE = (0.42/0.350500)**2,
        ninvjob_geometry = 'healpix_geometry',
        lmin_tlm = 30,
        lmin_elm = 30,
        lmin_blm = 30, #Supress all modes below this value, hacky version of OBD
        CENTRALNLEV_UKAMIN = 0.42,
        nlev_t = 0.42/np.sqrt(2),
        nlev_p = 0.42,
        nlev_dep = 10000.,
        inf = 1e4,
        mask = ('nlev', np.inf),
        rhits_normalised = ('/global/project/projectdirs/cmbs4/awg/lowellbb/reanalysis/mapphi_intermediate/s08b/masks/08b_rhits_positive_nonan.fits', np.inf),
        tpl = 'template_dense'
    ),
    qerec = DLENSALOT_Qerec(
        FILTER_QE = 'sepTP', # Change the following block only if other than sepTP for QE is desired
        CG_TOL = 1e-3,
        ninvjob_qe_geometry = 'healpix_geometry_qe',
        lmax_qlm = 4000,
        mmax_qlm = 4000,
        QE_LENSING_CL_ANALYSIS = False # Change the following block only if a full, Planck-like QE lensing power spectrum analysis is desired
    ),
    itrec = DLENSALOT_Itrec(
        FILTER = 'cinv_sepTP', # Change the following block only if other than cinv_t, cinv_p, ivfs filters are desired
        TOL = 3,
        lenjob_geometry = 'thin_gauss',
        lenjob_pbgeometry = 'pbdGeometry',
        ITERATOR = 'constmf', # Choose your iterator. Either pertmf or const_mf
        mfvar = '/global/cscratch1/sd/sebibel/cmbs4/08b_00_OBD_MF100_example/qlms_dd/simMF_k1p_p_135b0ca72339ac4eb092666cd7acb262a8ea2d30.fits',
        soltn_cond = lambda it: True,
    ),
    map_delensing = DLENSALOT_Mapdelensing(
        edges = 'ioreco',
        IMIN = 0,
        IMAX = 99,
        ITMAX = [10,12],
        droplist = np.array([]),
        base_mask = 'cmbs4/08b/caterinaILC_May12', # This mask is used to rotate ILC maps
        nlevels = [2, 5],
        lmax_cl = 2048,
        Cl_fid = 'ffp10'
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
)