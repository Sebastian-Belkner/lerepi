"""08d sims
    Params in Table 2-2 PBDR
    Each class defines a set of data which has been provided for delensing.
    For each class, define the directory, configuration, mask, ..
    Configuration of sims can be found in /config_survey
"""

import os
import numpy as np
from plancklens import utils #TODO switch this to lenscarf
import healpy as hp


class ILC_May2022:
    """ILC maps from Caterina Umilta on s08d May 2022 for the Chile configuration.
        These maps are multiplied with the weights used for the ILC
    """
    def __init__(self, fg, facunits=1e6, rhitsi=True, mask_suffix=None):
        """
        rhitsi is for reweighting maps by rhits value, if needed.
        """
        assert fg in ['00', '07']
        self.facunits = facunits
        self.fg = fg
        p_dset_dir =  '/project/projectdirs/cmbs4/awg/lowellbb/reanalysis/foreground_cleaned_maps/08d.%s_umilta_220502'%fg
        self.path = p_dset_dir + '/cmbs4_08d' + fg + '_cmb_b02_ellmin30_ellmax4050_map_2048_%04d.fits' # CMB + noise
        self.path_noise =   p_dset_dir + '/cmbs4_08d' + fg + '_noise_b02_ellmin30_ellmax4050_map_2048_%04d.fits'
        if mask_suffix is None:
            self.p_mask = p_dset_dir + '/ILC_mask_08d_smooth_30arcmin.fits' # Same mask as 06d
        else:
            self.p_mask = '/global/homes/s/sebibel/git/lerepi/lerepi/data/dc08/masks/mask_r%s.fits'%mask_suffix #/global/cscratch1/sd/sebibel/masks/cmbs4/masks/
        self.rhitsi = rhitsi
        self.nside_mask = 2048
    
    def hashdict(self):

        ret = {'rhits':self.rhitsi, 'sim_lib':'cmbs4_08d_ILC_%s'%self.fg, 'units':self.facunits, 'path2sim0':self.path%0}
        return ret


    def get_sim_pmap(self, idx):

        retq = np.nan_to_num(hp.read_map(self.path%idx, field=1)) * self.facunits
        retu = np.nan_to_num(hp.read_map(self.path%idx, field=2)) * self.facunits
        fac = 1. if not self.rhitsi else np.nan_to_num(hp.read_map(self.p_mask))
        return retq * utils.cli(fac), retu * utils.cli(fac)


    def get_mask(self):

        return hp.read_map(self.p_mask, field=0)
    
    
    def get_mask_path(self):
        
        return self.p_mask
    
    
    def get_nlev_mask(self, ratio):
        assert 0, 'Implement if needed'
        mask_loc = hp.read_map(self.p_mask, field=0)
        return np.where(rhits < 1. / ratio, 0., rhits)  *(~np.isnan(rhits))


    def get_noise_sim_pmap(self, idx):

        retq = np.nan_to_num(hp.read_map(self.path_noise%idx, field=1)) * self.facunits
        retu = np.nan_to_num(hp.read_map(self.path_noise%idx, field=2)) * self.facunits
        fac = 1. if not self.rhitsi else np.nan_to_num(hp.read_map(self.p_mask))
        return retq * utils.cli(fac), retu * utils.cli(fac)