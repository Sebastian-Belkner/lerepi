"""
90.91 sims

"""
import os
import numpy as np
import plancklens
from astropy.io import fits
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


def get_zbounds(rhits, hits_ratio=np.inf):
    """Cos-tht bounds for thresholded mask

    """
    pix = np.where(get_nlev_mask(rhits, hits_ratio))[0]
    tht, phi = hp.pix2ang(2048, pix)
    return np.cos(np.max(tht)), np.cos(np.min(tht))


def get_nlev_mask(rhits, ratio):
    """Mask built thresholding the relative hit counts map

        Note:
            Same as 06b

    """
    rhits = rhits
    mask = np.where(rhits < 1. / ratio, 0., 1.)  *(~np.isnan(rhits))
    return mask


def read_map(fn):
    return fits.open(fn)[0].data

def get_fidcls():
    """CMBs are the FFP10 ones

    """
    cls_path = os.path.join(os.path.dirname(os.path.abspath(plancklens.__file__)), 'data', 'cls')
    cl_unl = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))
    cl_len = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))
    return cl_unl, cl_len


class ILC_Matthieu_18:
    """ILC maps from Matthieu from 2018, nside 512

        These maps are multiplied with the weights used for the ILC #TODO

    """

    def __init__(self, fg, facunits=1e6, rhitsi=True):
        assert fg in ['91']
        self.facunits = facunits
        self.fg = fg

        p = "/project/projectdirs/pico/reanalysis/nilc/"
        self.path = p + '/py91_00%02d/NILC_PICO91_B_reso40acm.fits' # odd is r=0
        self.path_noise =   p + '/py91_00%02d/NILC_NOISE_PICO91_B_reso40acm.fits'
        self.rhitsi = rhitsi
        self.p2mask = "/global/homes/s/sebibel/data/mask/PR3vJan18_temp_lensingmask_gPR2_70_psPR2_143_COT2_smicadx12_smicapoldx12_psPR2_217_sz.fits.gz" #TODO
        # self.p2mask = "/project/projectdirs/pico/reanalysis/nilc/nilc_pico_mask.fits" #TODO


    def hashdict(self):
        ret = {'rhits':self.rhitsi, 'sim_lib':'pico_08b_ILC_%s'%self.fg, 'units':self.facunits, 'path2sim0':self.path%0}
        return ret


    def get_sim_pmap(self, idx):
        retq = np.nan_to_num(hp.read_map(self.path%str(int(2*idx+1)), field=1)) * self.facunits
        retu = np.nan_to_num(hp.read_map(self.path%str(int(2*idx+1)), field=2)) * self.facunits
        fac = 1. if not self.rhitsi else np.nan_to_num(hp.read_map(self.p2mask))
        return retq * utils.cli(fac), retu * utils.cli(fac)


    def get_noise_sim_pmap(self, idx):
        retq = np.nan_to_num(hp.read_map(self.path_noise%str(int(2*idx+1)), field=1)) * self.facunits
        retu = np.nan_to_num(hp.read_map(self.path_noise%str(int(2*idx+1)), field=2)) * self.facunits
        fac = 1. if not self.rhitsi else np.nan_to_num(hp.read_map(self.p2mask))
        return retq * utils.cli(fac), retu * utils.cli(fac)


class ILC_Matthieu_Dec21:
    """ILC maps from Mathieu on 90.91 Nov 2021

        No power above :math:`\ell = 2000` in these maps

    """

    def __init__(self, fg, facunits=1e6, rhitsi=True):
        assert fg in ['91']
        self.facunits = facunits
        self.fg = fg

        p = "/project/projectdirs/pico/reanalysis/nilc/ns2048"
        self.path_E = p + '/py91_00%02d/NILC_PICO91_E_reso8acm.fits' # odd is r=0
        self.path_B = p + '/py91_00%02d/NILC_PICO91_B_reso8acm.fits' # odd is r=0
        self.path_noise_E =   p + '/py91_00%02d/NILC_NOISE_PICO91_E_reso8acm.fits'
        self.path_noise_B =   p + '/py91_00%02d/NILC_NOISE_PICO91_B_reso8acm.fits'
        # p =  '/project/projectdirs/pico/data_xx.yy/90.00' # 08b.%s_umilta_210511/'%fg
        # self.path = p + '/pico_90_llcdm_AL0p03_f021_b38_ellmin00_map_2048_mc_%04d.fits' #TODO # CMB + noise 
        # self.path_noise =   p + '/pico_90_noise_f090_b10_ellmin00_map_2048_mc_%04d.fits' #TODO
        self.rhitsi = rhitsi
        #Lensing mask for now
        self.p2mask = "/global/homes/s/sebibel/data/mask/PR3vJan18_temp_lensingmask_gPR2_70_psPR2_143_COT2_smicadx12_smicapoldx12_psPR2_217_sz.fits.gz"
        #"/project/projectdirs/pico/reanalysis/compsepmaps/gnilc/small_mask_gnilc_90p91_fsky0-024.fits" #TODO


    def hashdict(self):
        ret = {'rhits':self.rhitsi, 'sim_lib':'pico_08b_ILC_%s'%self.fg, 'units':self.facunits, 'path2sim0':self.path%0}
        return ret


    def get_sim_pmap(self, idx):
        retE = np.nan_to_num(fits.open(self.path%str(int(2*idx+1)))[0].data) * self.facunits
        retB = np.nan_to_num(fits.open(self.path%str(int(2*idx+1)))[0].data) * self.facunits
        fac = 1. if not self.rhitsi else np.nan_to_num(hp.read_map(self.p2mask))
        return retE * utils.cli(fac), retB * utils.cli(fac)


    def get_noise_sim_pmap(self, idx):
        retE = np.nan_to_num(fits.open(self.path_noise_E%str(int(2*idx+1)))[0].data) * self.facunits
        retB = np.nan_to_num(fits.open(self.path_noise_B%str(int(2*idx+1)))[0].data) * self.facunits
        fac = 1. if not self.rhitsi else np.nan_to_num(hp.read_map(self.p2mask))
        return retE * utils.cli(fac), retB * utils.cli(fac)
    
    
class ILC_Seb_Nov21:
    """ILC maps from Seb on 90.91 Nov 2021, using SMICA/MV

        These maps are multiplied with the weights used for the ILC #TODO

    """

    def __init__(self, fg, nside, facunits=1e6, rhitsi=True):
        assert fg in ['91']
        self.facunits = facunits
        self.fg = fg
        p = "/global/cscratch1/sd/sebibel/compsep/pico/d90sim/lensmask/sim0"
        self.path = p + '/MapT_combined_SMICA_highell_bins_%04d'%nside+'_1500_2500_JC_%04d.npy'
        # self.path_noise =   p + '/ClN_non-separated_2048_4000_6000_JC_%04d.fits'
        pcmbs4 =  '/project/projectdirs/cmbs4/awg/lowellbb/reanalysis/foreground_cleaned_maps/08b.%s_umilta_210511/'%fg
        self.path_noise =   pcmbs4 + '/cmbs4_08b' + fg + '_noise_b02_ellmin30_ellmax4050_map_2048_%04d.fits' #TODO delete noise maps
        # p =  '/project/projectdirs/pico/data_xx.yy/90.00' # 08b.%s_umilta_210511/'%fg
        # self.path = p + '/pico_90_llcdm_AL0p03_f021_b38_ellmin00_map_2048_mc_%04d.fits' #TODO # CMB + noise 
        # self.path_noise =   p + '/pico_90_noise_f090_b10_ellmin00_map_2048_mc_%04d.fits' #TODO
        
        self.rhitsi = rhitsi
        self.p2mask = "/global/homes/s/sebibel/data/mask/PR3vJan18_temp_lensingmask_gPR2_70_psPR2_143_COT2_smicadx12_smicapoldx12_psPR2_217_sz.fits.gz" #TODO


    def hashdict(self):
        ret = {'rhits':self.rhitsi, 'sim_lib':'pico_08b_ILC_%s'%self.fg, 'units':self.facunits, 'path2sim0':self.path%0}
        return ret


    def get_sim_pmap(self, idx):
        retq = np.nan_to_num(np.load(self.path%idx))[1] * self.facunits
        retu = np.nan_to_num(np.load(self.path%idx))[2] * self.facunits
        fac = 1. if not self.rhitsi else np.nan_to_num(hp.read_map(self.p2mask))
        return retq * utils.cli(fac), retu * utils.cli(fac)

        # retq = np.nan_to_num(hp.read_map(self.path%idx, field=1)) * self.facunits
        # retu = np.nan_to_num(hp.read_map(self.path%idx, field=2)) * self.facunits
        # fac = 1. if not self.rhitsi else np.nan_to_num(hp.read_map(self.p2mask))
        # return retq * utils.cli(fac), retu * utils.cli(fac)

    def get_noise_sim_pmap(self, idx):
        retq = np.nan_to_num(np.load(self.path_noise%idx))[1] * self.facunits
        retu = np.nan_to_num(np.load(self.path_noise%idx))[2] * self.facunits
        fac = 1. if not self.rhitsi else np.nan_to_num(hp.read_map(self.p2mask))
        return retq * utils.cli(fac), retu * utils.cli(fac)


class ILC_Pico_2018:
    """
    Dummy class, not sure where to find the data...
    ILC maps from unknown on 90.91 from 2018. 

    """


    def __init__(self, fg, facunits=1e6, rhitsi=True):
        assert 0, 'Dummy class, implement if needed.'
        assert fg in ['91']
        self.facunits = facunits
        self.fg = fg
        p = "/global/cscratch1/sd/sebibel/compsep/pico/d90sim/lensmask/sim0"
        self.path = p + '/MapT_combined_SMICA_highell_bins_2048_1500_2500_JC_%04d.npy'
        # self.path_noise =   p + '/ClN_non-separated_2048_4000_6000_JC_%04d.fits'
        pcmbs4 =  '/project/projectdirs/cmbs4/awg/lowellbb/reanalysis/foreground_cleaned_maps/08b.%s_umilta_210511/'%fg
        self.path_noise =   pcmbs4 + '/cmbs4_08b' + fg + '_noise_b02_ellmin30_ellmax4050_map_2048_%04d.fits' #TODO delete noise maps
        # p =  '/project/projectdirs/pico/data_xx.yy/90.00' # 08b.%s_umilta_210511/'%fg
        # self.path = p + '/pico_90_llcdm_AL0p03_f021_b38_ellmin00_map_2048_mc_%04d.fits' #TODO # CMB + noise 
        # self.path_noise =   p + '/pico_90_noise_f090_b10_ellmin00_map_2048_mc_%04d.fits' #TODO
        
        self.rhitsi = rhitsi
        self.p2mask = "/global/homes/s/sebibel/data/mask/PR3vJan18_temp_lensingmask_gPR2_70_psPR2_143_COT2_smicadx12_smicapoldx12_psPR2_217_sz.fits.gz" #TODO


    def hashdict(self):
        ret = {'rhits':self.rhitsi, 'sim_lib':'pico_08b_ILC_%s'%self.fg, 'units':self.facunits, 'path2sim0':self.path%0}
        return ret


    def get_sim_pmap(self, idx):
        retq = np.nan_to_num(np.load(self.path%idx))[1] * self.facunits
        retu = np.nan_to_num(np.load(self.path%idx))[2] * self.facunits
        fac = 1. if not self.rhitsi else np.nan_to_num(hp.read_map(self.p2mask))
        return retq * utils.cli(fac), retu * utils.cli(fac)

        # retq = np.nan_to_num(hp.read_map(self.path%idx, field=1)) * self.facunits
        # retu = np.nan_to_num(hp.read_map(self.path%idx, field=2)) * self.facunits
        # fac = 1. if not self.rhitsi else np.nan_to_num(hp.read_map(self.p2mask))
        # return retq * utils.cli(fac), retu * utils.cli(fac)

    def get_noise_sim_pmap(self, idx):
        retq = np.nan_to_num(np.load(self.path_noise%idx))[1] * self.facunits
        retu = np.nan_to_num(np.load(self.path_noise%idx))[2] * self.facunits
        fac = 1. if not self.rhitsi else np.nan_to_num(hp.read_map(self.p2mask))
        return retq * utils.cli(fac), retu * utils.cli(fac)