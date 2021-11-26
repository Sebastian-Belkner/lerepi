"""
90.91 sims

"""
import os
import numpy as np
import plancklens
from plancklens import utils
import healpy as hp


def get_beam(freq):

    return {21: 38.4, 25: 32.0,  30: 28.3,  36: 23.6,  43: 22.2,  52: 18.4,  
        62: 12.8,  75: 10.7,  90: 9.5,  108: 7.9,  129: 7.4,  155: 6.2, 
        186: 4.3,  223: 3.6,  268: 3.2,  321: 2.6,  385: 2.5,  462: 2.1, 
        555: 1.5,  666: 1.3,  799: 1.1 
    }[freq]


def get_nlevp(freq):
                      
    return {21: 16.9, 25: 13.0,  30: 8.7,  36: 5.6,  43: 5.6,  52: 4.0,  
        62: 3.8, 75: 3.0, 90: 2.0, 108: 1.6, 129: 1.5,  155: 1.3, 
        186: 2.8,  223: 3.2,  268: 2.2,  321: 3.0,  385: 3.2,  462: 6.4, 
        555: 32.4,  666: 125,  799: 740
    }[freq]


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
    """ILC maps from Clem on 90.91 Nov 2021

        These maps are multiplied with the weights used for the ILC #TODO

    """


    def __init__(self, fg, facunits=1e6, rhitsi=True):
        assert fg in ['91']
        self.facunits = facunits
        self.fg = fg

        p = "/project/projectdirs/pico/reanalysis/compsepmaps"
        self.path = p + '/pico_90_combined_map_2048_mc_%04d.fits'
        self.path_noise =   p + '/pico_90_noise_combined_map_2048_mc_%04d.fits'
        # p =  '/project/projectdirs/pico/data_xx.yy/90.00' # 08b.%s_umilta_210511/'%fg
        # self.path = p + '/pico_90_llcdm_AL0p03_f021_b38_ellmin00_map_2048_mc_%04d.fits' #TODO # CMB + noise 
        # self.path_noise =   p + '/pico_90_noise_f090_b10_ellmin00_map_2048_mc_%04d.fits' #TODO
        
        self.rhitsi = rhitsi
        self.p2mask = "/project/projectdirs/pico/reanalysis/compsepmaps/gnilc/small_mask_gnilc_90p91_fsky0-024.fits" #TODO


    def hashdict(self):
        ret = {'rhits':self.rhitsi, 'sim_lib':'pico_08b_ILC_%s'%self.fg, 'units':self.facunits, 'path2sim0':self.path%0}
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