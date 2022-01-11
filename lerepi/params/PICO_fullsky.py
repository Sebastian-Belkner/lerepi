"""Iterative reconstruction for masked polarization CMB data

    On idealized, full-sky maps, with homogeneous noise
    The difference to 'cmbs4wide_planckmask' is usage of a simpler filter instance, which works in harmonic space


FIXME's :
    plancklens independent QEs ?
    check of invertibility at very first step

"""
import os
from os.path import join as opj
import numpy as np

import plancklens

from plancklens import utils, qresp, qest
from plancklens.qcinv import cd_solve
from plancklens.filt import filt_simple

from lenscarf import remapping, utils_scarf
from lenscarf.iterators import cs_iterator as scarf_iterator, steps
from lenscarf.utils import cli
from lenscarf.utils_hp import almxfl, alm_copy
from lenscarf.opfilt.opfilt_iso_ee_wl import alm_filter_nlev_wl

from lerepi import sims_90


suffix = 'PICO_idealized' # descriptor to distinguish this parfile from others...
TEMP =  opj(os.environ['SCRATCH'], 'lenscarfrecs', suffix)

sims      = sims_90.NILC_idealE() # Matthieu NILC B-modes with FFP10 E-modes and Gaussian E-noise

lmax_ivf, mmax_ivf = (2000, 2000)

nlev_p = np.sqrt(sims.clnoise) * (60 * 180 / np.pi) # Here we will use harmonic space iterators with colored noise
nlev_t = np.sqrt(sims.clnoise) * (60 * 180 / np.pi) / np.sqrt(2.) # irrelevant

lmin_tlm, lmin_elm, lmin_blm = (30, 10, 200) # The fiducial transfer functions are set to zero below these lmins
# for delensing useful to cut much more B. It can also help since the cg inversion does not have to reconstruct those.

lmax_qlm, mmax_qlm = (2500, 2500) # Lensing map is reconstructed down to this lmax and mmax
# NB: the QEs from plancklens does not support mmax != lmax, but the MAP pipeline does
lmax_unl, mmax_unl = (2500, 2500) # Delensed CMB is reconstructed down to this lmax and mmax


#----------------- pixelization and geometry info for the input maps and the MAP pipeline and for lensing operations

zbounds_len = (-1.,1.) # Outside of these bounds the reconstructed maps are assumed to be zero
pb_ctr, pb_extent = (0., 2 * np.pi) # Longitude cuts, if any, in the form (center of patch, patch extent)
lenjob_geometry = utils_scarf.Geom.get_thingauss_geometry(lmax_unl, 2, zbounds=zbounds_len)
lenjob_pbgeometry =utils_scarf.pbdGeometry(lenjob_geometry, utils_scarf.pbounds(pb_ctr, pb_extent))
lensres = 1.7  # Deflection operations will be performed at this resolution
Lmin = 2 # The reconstruction of all lensing multipoles below that will not be attempted
stepper = steps.nrstep(lmax_qlm, mmax_qlm, val=0.5) # handler of the size steps in the MAP BFGS iterative search
mc_sims_mf_it0 = np.array([]) # sims to use to build the very first iteration mean-field (QE mean-field) Here 0 since idealized


# Multigrid chain descriptor
chain_descrs = lambda lmax_sol, cg_tol : [[0, ["diag_cl"], lmax_sol, 2048, np.inf, cg_tol, cd_solve.tr_cg, cd_solve.cache_mem()]]
libdir_iterators = lambda qe_key, simidx, version: opj(TEMP,'%s_sim%04d'%(qe_key, simidx) + version)
#------------------

# Fiducial CMB spectra for QE and iterative reconstructions
# (here we use very lightly suboptimal lensed spectra QE weights)
cls_path = opj(os.path.dirname(plancklens.__file__), 'data', 'cls')
cls_unl = utils.camb_clfile(opj(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))
cls_len = utils.camb_clfile(opj(cls_path, 'FFP10_wdipole_lensedCls.dat'))

# Fiducial model of the transfer function
transf_tlm   =  sims.get_transf(lmax_ivf) * (np.arange(lmax_ivf + 1) >= lmin_tlm)
transf_elm   =  sims.get_transf(lmax_ivf) * (np.arange(lmax_ivf + 1) >= lmin_elm)
transf_blm   =  sims.get_transf(lmax_ivf) * (np.arange(lmax_ivf + 1) >= lmin_blm)
transf_d = {'t':transf_tlm, 'e':transf_elm, 'b':transf_blm}

# Isotropic approximation to the filtering (used eg for response calculations)
ftl =  cli(cls_len['tt'][:lmax_ivf + 1] + (nlev_t / 180 / 60 * np.pi) ** 2 * cli(transf_tlm ** 2)) * (transf_tlm > 0)
fel =  cli(cls_len['ee'][:lmax_ivf + 1] + (nlev_p / 180 / 60 * np.pi) ** 2 * cli(transf_elm ** 2)) * (transf_elm > 0)
fbl =  cli(cls_len['bb'][:lmax_ivf + 1] + (nlev_p / 180 / 60 * np.pi) ** 2 * cli(transf_blm ** 2)) * (transf_blm > 0)

# Same using unlensed spectra (used for unlensed response used to initiate the MAP curvature matrix)
ftl_unl =  cli(cls_unl['tt'][:lmax_ivf + 1] + (nlev_t / 180 / 60 * np.pi) ** 2 * cli(transf_tlm ** 2)) * (transf_tlm > 0)
fel_unl =  cli(cls_unl['ee'][:lmax_ivf + 1] + (nlev_p / 180 / 60 * np.pi) ** 2 * cli(transf_elm ** 2)) * (transf_elm > 0)
fbl_unl =  cli(cls_unl['bb'][:lmax_ivf + 1] + (nlev_p / 180 / 60 * np.pi) ** 2 * cli(transf_blm ** 2)) * (transf_blm > 0)

# -------------------------
transf_dat =  sims.get_transf(2000) # (taking here full FFP10 cmb's which are given to 4096)

# Makes the simulation library consistent with the zbounds
sims_MAP  = sims
# -------------------------

ivfs   = filt_simple.library_fullsky_alms_sepTP(opj(TEMP, 'ivfs'), sims, transf_d, cls_len, ftl, fel, fbl, cache=True)
qlms_dd = qest.library_sepTP(opj(TEMP, 'qlms_dd'), ivfs, ivfs,   cls_len['te'], 2048, lmax_qlm=lmax_qlm)


def get_itlib(k:str, simidx:int, version:str, cg_tol:float):
    """Return iterator instance for simulation idx and qe_key type k

        Args:
            k: 'p_p' for Pol-only, 'ptt' for T-only, 'p_eb' for EB-only, etc
            simidx: simulation index to build iterative lensing estimate on
            version: string to use to test variants of the iterator with otherwise the same parfile
                     (here if 'noMF' is in version, will not use any mean-fied at the very first step)
            cg_tol: tolerance of conjugate-gradient filter

    """
    assert k in ['p_eb', 'p_p'], k
    libdir_iterator = libdir_iterators(k, simidx, version)
    if not os.path.exists(libdir_iterator):
        os.makedirs(libdir_iterator)
    tr = int(os.environ.get('OMP_NUM_THREADS', 8))
    cpp = np.copy(cls_unl['pp'][:lmax_qlm + 1])

    # QE mean-field fed in as constant piece in the iteration steps:
    mf_sims = np.unique(mc_sims_mf_it0 if not 'noMF' in version else np.array([]))
    mf0 = qlms_dd.get_sim_qlm_mf(k, mf_sims)  # Mean-field to subtract on the first iteration:
    if simidx in mf_sims:  # We dont want to include the sim we consider in the mean-field...
        mf0 = (mf0 - qlms_dd.get_sim_qlm(k, int(simidx)) / len(mf_sims)) / (len(mf_sims) - 1)

    path_plm0 = opj(libdir_iterator, 'phi_plm_it000.npy')
    if not os.path.exists(path_plm0):
        # We now build the Wiener-filtered QE here since not done already
        plm0  = qlms_dd.get_sim_qlm(k, int(simidx))  #Unormalized quadratic estimate:
        plm0 -= mf0  # MF-subtracted unnormalized QE
        # Isotropic normalization of the QE
        R = qresp.get_response(k, lmax_ivf, 'p', cls_len, cls_len, {'e': fel, 'b': fbl, 't':ftl}, lmax_qlm=lmax_qlm)[0]
        # Isotropic Wiener-filter (here assuming for simplicity N0 ~ 1/R)
        WF = cpp * utils.cli(cpp + utils.cli(R))
        plm0 = alm_copy(plm0,  None, lmax_qlm, mmax_qlm) # Just in case the QE and MAP mmax'es were not consistent
        almxfl(plm0, utils.cli(R), mmax_qlm, True) # Normalized QE
        almxfl(plm0, WF, mmax_qlm, True)           # Wiener-filter QE
        np.save(path_plm0, plm0)

    plm0 = np.load(path_plm0)
    R_unl = qresp.get_response(k, lmax_ivf, 'p', cls_unl, cls_unl,  {'e': fel_unl, 'b': fbl_unl, 't':ftl_unl}, lmax_qlm=lmax_qlm)[0]
    if k in ['p_p']:
        mf_resp = qresp.get_mf_resp(k, cls_unl, {'ee': fel_unl, 'bb': fbl_unl}, lmax_ivf, lmax_qlm)[0]
    else:
        print('*** mf_resp not implemented for key ' + k, ', setting it to zero')
        mf_resp = np.zeros(lmax_qlm + 1, dtype=float)
    # Lensing deflection field instance (initiated here with zero deflection)
    ffi = remapping.deflection(lenjob_pbgeometry, lensres, np.zeros_like(plm0), mmax_qlm, tr, tr)
    if k in ['p_p', 'p_eb']:
        wee = k == 'p_p' # keeps or not the EE-like terms in the generalized QEs
        assert np.all(transf_elm == transf_blm), 'This is not supported by the alm_filter_nlev_wl (but easy to fix)'
        # Here multipole cuts are set by the transfer function (those with 0 are not considered)
        filtr = alm_filter_nlev_wl(nlev_p, ffi, transf_elm, (lmax_unl, mmax_unl), (lmax_ivf, mmax_ivf),
                                   wee=wee, transf_b=transf_blm, nlev_b=nlev_p)
        # dat maps now given in harmonic in this idealized configuration
        datmaps = np.array(sims_MAP.get_sim_pmap(int(simidx)))
    else:
        assert 0
    k_geom = filtr.ffi.geom # Customizable Geometry for position-space operations in calculations of the iterated QEs etc
    # Sets to zero all L-modes below Lmin in the iterations:
    cpp[:Lmin] *= 0.
    almxfl(plm0, cpp > 0, mmax_qlm, True)
    iterator = scarf_iterator.iterator_pertmf(libdir_iterator, 'p', (lmax_qlm, mmax_qlm), datmaps,
            plm0, mf_resp, R_unl, cpp, cls_unl, filtr, k_geom, chain_descrs(lmax_unl, cg_tol), stepper
            ,mf0=mf0)
    return iterator

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='test iterator full-sky with pert. resp.')
    parser.add_argument('-k', dest='k', type=str, default='p_p', help='rec. type')
    parser.add_argument('-itmax', dest='itmax', type=int, default=-1, help='maximal iter index')
    parser.add_argument('-tol', dest='tol', type=float, default=5., help='-log10 of cg tolerance default')
    parser.add_argument('-imin', dest='imin', type=int, default=-1, help='minimal sim index')
    parser.add_argument('-imax', dest='imax', type=int, default=-1, help='maximal sim index')
    parser.add_argument('-v', dest='v', type=str, default='', help='iterator version')


    args = parser.parse_args()
    tol_iter   = lambda it : 10 ** (- args.tol) # tolerance a fct of iterations ?
    soltn_cond = lambda it: True # Uses (or not) previous E-mode solution as input to search for current iteration one

    from plancklens.helpers import mpi
    mpi.barrier = lambda : 1 # redefining the barrier (Why ? )
    from lenscarf.iterators.statics import rec as Rec
    jobs = []
    for idx in np.arange(args.imin, args.imax + 1):
        lib_dir_iterator = libdir_iterators(args.k, idx, args.v)
        if Rec.maxiterdone(lib_dir_iterator) < args.itmax:
            jobs.append(idx)

    for idx in jobs[mpi.rank::mpi.size]:
        lib_dir_iterator = libdir_iterators(args.k, idx, args.v)
        if args.itmax >= 0 and Rec.maxiterdone(lib_dir_iterator) < args.itmax:
            itlib = get_itlib(args.k, idx, args.v, 1.)
            for i in range(args.itmax + 1):
                print("****Iterator: setting cg-tol to %.4e ****"%tol_iter(i))
                print("****Iterator: setting solcond to %s ****"%soltn_cond(i))

                itlib.chain_descr  = chain_descrs(lmax_unl, tol_iter(i))
                itlib.soltn_cond   = soltn_cond(i)
                print("doing iter " + str(i))
                itlib.iterate(i, 'p')