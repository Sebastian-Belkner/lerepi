"""This module contains utilities for forecast / predictions


"""
import os
import numpy as np

import plancklens
from plancklens import utils

def BBsqueeze(clee_unl, clpp):
    """Returns large scale squeezed BB limit predictions for EE and pp input

        See Lewis & Challinor review Eq. 5.34
    """
    lmax = min(len(clee_unl), len(clpp)) - 1
    ls = np.arange(lmax + 1, dtype=float)
    return np.sum( ls ** 3 * clpp[:lmax + 1] * ls ** 2 * clee_unl[:lmax + 1] ) / (4. * np.pi)

if __name__ == '__main__':
    """This calculates and plot predictions for NILC_idealE sims
    
    
    """
    import pylab as pl
    from plancklens import n0s
    from lerepi import sims_90
    #FIXME: this does not differentiate lmin_elm and lmin_blm
    lmin_cmb, lmax_cmb, lmax_qlm = (10, 2000, 2500)
    sims = sims_90.NILC_idealE()  # Matthieu NILC B-modes with FFP10 E-modes and Gaussian E-noise
    nlev_p = np.sqrt(sims.clnoise) * (60 * 180 / np.pi)  # Here we will use harmonic space iterators with colored noise
    nlev_t = np.sqrt(sims.clnoise) * (60 * 180 / np.pi) / np.sqrt(2.)  # irrelevant


    cls_path = os.path.join(os.path.dirname(os.path.abspath(plancklens.__file__)), 'data', 'cls')
    cls_len = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))
    cls_unl = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))

    (N0s_it, _, delcls_fid, _) = n0s.get_N0_iter('p_p', nlev_t, nlev_p, 8., cls_unl, 10, 2000, 6, lmax_qlm=2500, ret_delcls=True)

    BB = BBsqueeze(cls_unl['ee'], cls_unl['pp'])
    ls  = np.arange(2, 2500)
    ee_templ = np.copy(cls_unl['ee'][:lmax_cmb + 1])
    ee_templ[:lmin_cmb] *= 0

    ls_mean = np.arange(30, 190)
    for N0, delcl in zip([N0s_it[0], N0s_it[-1]], [delcls_fid[1], delcls_fid[-1]]):
        pl.plot(ls, 1e7 * ls ** 2 * (ls + 1) ** 2 * N0[ls] / 2 / np.pi)
        rho2 = np.zeros_like(cls_unl['pp'])

        rho2[:lmax_qlm + 1] = cls_unl['pp'][:lmax_qlm + 1] * utils.cli(cls_unl['pp'][:lmax_qlm + 1] + N0[:lmax_qlm + 1])
        BB_res = BBsqueeze(np.copy(cls_unl['ee']), cls_unl['pp'] * (1. - rho2))
        print('removed B power in percent  %.1f' % (100 * (1. - BB_res / BB)))
        BB_temp = BBsqueeze(ee_templ, cls_unl['pp'] * rho2)
        BB_res_camb = np.mean(delcl['bb'][ls_mean])
        # 3 very slightly different ways to calculate the residual B:
        print('%.2f' % (1 - BB_res / BB), '%.2f' % (1. - (BB - BB_temp) / BB), '%.2f' % (1 - BB_res_camb / BB), )
    pl.loglog(ls, 1e7 * ls ** 2 * (ls + 1) ** 2 * cls_unl['pp'][ls], c='k', label=r'$C_L^{\phi\phi, \rm fid}$')
    pl.ylabel(r'$10^7 \cdot L^2 (L + 1)^2 C_L^{\rm \phi \phi} /2\pi$', fontsize=14)
    pl.xlabel(r'$L$', fontsize=14)
    pl.legend(fontsize=14)