"""
90.91 sims

"""
import os
import numpy as np
import plancklens
from plancklens import utils
import healpy as hp
import cmbs4

#TODO
def get_beam(freq):
    return {20:11.0, 27:8.4, 39:5.8, 93:2.5, 145:1.6, 225:1.1, 278:1.}[freq]

#TODO
def get_nlevp(freq):
    return {20:13.6, 27:6.5, 39:4.15, 93:0.63, 145:0.59, 225:1.83, 278:4.34}[freq]

#TODO
def get_PBDR_noise_cl(freq, lmax):
    lknee = dict()
    a_P = dict()
    for f in [20, 27, 39, 93]:
        lknee[f] = 150
        a_P[f] = -2.7
    a_P[93] = -2.6
    for f in [145, 225, 278]:
        lknee[f] = 200
        a_P[f] = -2.2
    nw = get_nlevp(freq)
    cl_ee = (utils.cli(np.arange(lmax + 1) / lknee[freq]) ** (-a_P[freq]) * nw ** 2 + nw ** 2) * (np.pi / 180 / 60) ** 2
    return cl_ee, cl_ee.copy()

#TODO
def get_crudeILC_noise_cl(fg, lmax, output_beam, freqs=(95, 145), ret_coeffs=False, spl_coeffs=False):
    """Builds coefficients and noise curves for harmonic ILC from empirical spectra

        If output_beam is set, the noise curves corresponds to the CMB beamed with that width (in amin)
        The coefficients always refer to the deconvolved CMB

    """
    assert fg in ['nofg', '00', '07', '09'], fg
    Nf = len(freqs)
    Nee = np.zeros((lmax + 1, Nf, Nf))
    Nbb = np.zeros((lmax + 1, Nf, Nf))

    for ifr, freq1 in enumerate(freqs):
        transf1 = hp.gauss_beam(get_beam(freq1) / 180 / 60 * np.pi, lmax=lmax)
        for ifr2, freq2 in enumerate(freqs):
            transf2 = hp.gauss_beam(get_beam(freq2) / 180 / 60 * np.pi, lmax=lmax)
            if fg != 'nofg':
                fg_clee = get_beamdfgd_pcls(min(freq1, freq2), max(freq1, freq2), fg)[0, :lmax + 1]
                fg_clbb = get_beamdfgd_pcls(min(freq1, freq2), max(freq1, freq2), fg)[1, :lmax + 1]
                Nee[:len(fg_clee), ifr, ifr2] += fg_clee * utils.cli(transf1 * transf2)[:len(fg_clee)]
                Nbb[:len(fg_clbb), ifr, ifr2] += fg_clbb * utils.cli(transf1 * transf2)[:len(fg_clbb)]
            if freq2 == freq1:
                een, bbn = get_PBDR_noise_cl(freq1, lmax)
                Nee[:, ifr, ifr2] += een * utils.cli(transf1 * transf2)
                Nbb[:, ifr, ifr2] += bbn * utils.cli(transf1 * transf2)

    Nee_ILC = 1. / np.sum(np.linalg.pinv(Nee), axis=(1, 2))  # ILC noise curves
    Nbb_ILC = 1. / np.sum(np.linalg.pinv(Nbb), axis=(1, 2))
    if output_beam > 0:
        transf2 = hp.gauss_beam(output_beam / 180 / 60 * np.pi, lmax=lmax) ** 2
        Nee_ILC *= transf2
        Nbb_ILC *= transf2
    if ret_coeffs:
        eea_ICL = np.sum(np.linalg.pinv(Nee), axis=2).transpose() * Nee_ILC  # ILC coefficients
        bba_ICL = np.sum(np.linalg.pinv(Nbb), axis=2).transpose() * Nbb_ILC
        if spl_coeffs and fg != 'nofg':
            from scipy.interpolate import UnivariateSpline as spl
            ls = np.arange(2, lmax + 1)
            for i in range(Nf - 1):
                eea_ICL[i, ls] = spl(ls, eea_ICL[i][ls], k=2, s=1)(ls * 1.)
                bba_ICL[i, ls] = spl(ls, bba_ICL[i][ls], k=2, s=1)(ls * 1.)
            eea_ICL[Nf - 1] = 1. - np.sum(eea_ICL[:Nf-1], axis=0)
            bba_ICL[Nf - 1] = 1. - np.sum(bba_ICL[:Nf-1], axis=0)
        return Nee_ILC, Nbb_ILC, eea_ICL, bba_ICL
    return Nee_ILC, Nbb_ILC

#TODO
def get_beamdfgd_pcls(freq1, freq2, fg, mask_nlev=2.):
    """EE BB EB BE foreground (cross)spectra, inclusive of transfer function

    """
    print("08b: using 06b foregrounds empirical spectra")
    if freq1 in [93, 95]: freq1 = 95
    if freq2 in [93, 95]: freq2 = 95
    suf = 'fg%s_pcls_%sGHzx%sGHz_mnlev_%s.dat'%(fg, freq1, freq2, int(100 * mask_nlev))
    p = os.path.join(os.path.dirname(os.path.abspath(cmbs4.__file__)), 'data', 's06b', suf)
    return np.loadtxt(p).transpose()

#TODO
def get_zbounds(hits_ratio=np.inf):
    """Cos-tht bounds for thresholded mask

    """
    pix = np.where(get_nlev_mask(hits_ratio))[0]
    tht, phi = hp.pix2ang(2048, pix)
    return np.cos(np.max(tht)), np.cos(np.min(tht))

#TODO
def get_nlev_mask(ratio):
    """Mask built thresholding the relative hit counts map

        Note:
            Same as 06b

    """
    rhits = hp.read_map('/project/projectdirs/cmbs4/awg/lowellbb/expt_xx/08b/rhits/n2048.fits')
    mask = np.where(rhits < 1. / ratio, 0., 1.)  *(~np.isnan(rhits))
    return mask

#TODO
def get_fidcls():
    cls_path = os.path.join(os.path.dirname(os.path.abspath(plancklens.__file__)), 'data', 'cls')
    cl_unl = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))
    cl_len = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))
    return cl_unl, cl_len


class ILC_Clem_Nov21:
    """ILC maps from C Umilta on s06b May 12 2021

        These maps are multiplied with the weights used for the ILC

    """
    #TODO paths
    def __init__(self, fg, facunits=1e6, rhitsi=True):
        assert fg in ['91']
        self.facunits = facunits
        self.fg = fg
        p =  '/project/projectdirs/pico/reanalysis/compsepmaps' # 08b.%s_umilta_210511/'%fg
        self.path = p + '/cmbs4_08b' + fg + '_cmb_b02_ellmin30_ellmax4050_map_2048_%04d.fits' # CMB + noise
        self.path_noise =   p + '/cmbs4_08b' + fg + '_noise_b02_ellmin30_ellmax4050_map_2048_%04d.fits'
        self.rhitsi=rhitsi
        self.p2mask = p + '/ILC_mask_08b_smooth_30arcmin.fits' # Same mask as 06b


    def hashdict(self):
        ret = {'rhits':self.rhitsi, 'sim_lib':'cmbs4_08b_ILC_%s'%self.fg, 'units':self.facunits, 'path2sim0':self.path%0}
        return ret


    def get_sim_pmap(self, idx):
        retq = np.nan_to_num(hp.read_map(self.path%idx, field=1)) * self.facunits
        retu = np.nan_to_num(hp.read_map(self.path%idx, field=2)) * self.facunits
        fac = 1. if not self.rhitsi else np.nan_to_num(hp.read_map(self.p2mask))
        return retq * utils.cli(fac), retu * utils.cli(fac)


    def get_noise_sim_pmap(self, idx):
        retq = np.nan_to_num(hp.read_map(self.path_noise%idx, field=1)) * self.facunits
        retu = np.nan_to_num(hp.read_map(self.path_noise%idx, field=2)) * self.facunits
        fac = 1. if not self.rhitsi else np.nan_to_num(hp.read_map(self.p2mask))
        return retq * utils.cli(fac), retu * utils.cli(fac)