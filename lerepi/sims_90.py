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
    assert 0, "To be implemented"
    return {20:11.0, 27:8.4, 39:5.8, 93:2.5, 145:1.6, 225:1.1, 278:1.}[freq]

#TODO
def get_nlevp(freq):
    assert 0, "To be implemented"
    return {20:13.6, 27:6.5, 39:4.15, 93:0.63, 145:0.59, 225:1.83, 278:4.34}[freq]


def get_zbounds(hits_ratio=np.inf):
    """Cos-tht bounds for thresholded mask

    """
    pix = np.where(get_nlev_mask(hits_ratio))[0]
    tht, phi = hp.pix2ang(2048, pix)
    return np.cos(np.max(tht)), np.cos(np.min(tht))


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
    assert 0, "To be implemented"
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
        assert 0, "To be implemented"
        assert fg in ['91']
        self.facunits = facunits
        self.fg = fg
        p =  '/project/projectdirs/pico/reanalysis/compsepmaps' # 08b.%s_umilta_210511/'%fg
        self.path = p + '/cmbs4_08b' + fg + '_cmb_b02_ellmin30_ellmax4050_map_2048_%04d.fits' # CMB + noise
        self.path_noise =   p + '/cmbs4_08b' + fg + '_noise_b02_ellmin30_ellmax4050_map_2048_%04d.fits'
        self.rhitsi=rhitsi
        self.p2mask = p + '/ILC_mask_08b_smooth_30arcmin.fits'


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
