import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import glob, os
import re
import netCDF4
import copy
import string
import h5py
from typing import List
from tqdm import tqdm
import zarr

class data_utils:
    def __init__(self,
                 grid_info,
                 input_mean,
                 input_max,
                 input_min,
                 output_scale,
                 qinput_log = False,
                 normalize = True,
                 input_abbrev = 'mlis',
                 output_abbrev = 'mlos',
                 save_zarr=False,
                 save_h5=False,
                 save_npy=True,
                 cpuonly=False):
        if cpuonly:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.input_abbrev = input_abbrev
        self.output_abbrev = output_abbrev
        self.data_path = None
        self.save_zarr = save_zarr
        self.save_h5 = save_h5
        self.save_npy = save_npy
        self.input_vars = []
        self.target_vars = []
        self.input_feature_len = None
        self.target_feature_len = None
        self.grid_info = grid_info
        self.level_name = 'lev'
        self.sample_name = 'sample'
        self.num_levels = len(self.grid_info['lev'])
        self.num_latlon = len(self.grid_info['ncol']) # number of unique lat/lon grid points
        # make area-weights
        self.grid_info['area_wgt'] = self.grid_info['area']/self.grid_info['area'].mean(dim = 'ncol')
        self.area_wgt = self.grid_info['area_wgt'].values
        self.grid_info_area = self.grid_info['area'].values
        # map ncol to nsamples dimension
        # to_xarray = {'area_wgt':(self.sample_name,np.tile(self.grid_info['area_wgt'], int(n_samples/len(self.grid_info['ncol']))))}
        # to_xarray = xr.Dataset(to_xarray)
        self.input_mean = input_mean
        self.input_max = input_max
        self.input_min = input_min
        self.output_scale = output_scale
        self.normalize = normalize
        self.lats = self.grid_info['lat'].values
        self.lons = self.grid_info['lon'].values
        self.unique_lats, self.lats_indices = np.unique(self.lats, return_index=True)
        self.unique_lons, self.lons_indices = np.unique(self.lons, return_index=True)
        self.sort_lat_key = np.argsort(self.grid_info['lat'].values[np.sort(self.lats_indices)])
        self.sort_lon_key = np.argsort(self.grid_info['lon'].values[np.sort(self.lons_indices)])
        self.indextolatlon = {i: (self.grid_info['lat'].values[i%self.num_latlon], self.grid_info['lon'].values[i%self.num_latlon]) for i in range(self.num_latlon)}
        self.qinput_log = qinput_log

        def find_keys(dictionary, value):
            keys = []
            for key, val in dictionary.items():
                if val[0] == value:
                    keys.append(key)
            return keys
        indices_list = []
        for lat in self.unique_lats:
            indices = find_keys(self.indextolatlon, lat)
            indices_list.append(indices)
        indices_list.sort(key = lambda x: x[0])
        self.lat_indices_list = indices_list

        if self.num_latlon == 384:
            self.lat_bins = np.arange(-90, 91, 10)  # Create edges for 10 degree bins
            self.lat_bin_mids = self.lat_bins[:-1] + 5
            self.lat_bin_indices = np.digitize(self.lats, self.lat_bins) - 1
            self.lat_bin_dict = {lat_bin: self.lat_bin_indices == lat_bin for lat_bin in range(len(self.lat_bins) - 1)}
            self.lat_bin_area_sums = np.array([np.sum(self.grid_info_area[self.lat_bin_dict[lat_bin]]) for lat_bin in self.lat_bin_dict.keys()])
            self.lat_bin_area_divs = np.array([1/self.lat_bin_area_sums[bin_index] for bin_index in self.lat_bin_indices])

        self.hyam = self.grid_info['hyam'].values
        self.hybm = self.grid_info['hybm'].values
        self.p0 = 1e5 # code assumes this will always be a scalar
        self.ps_index = None

        self.pressure_grid_train = None
        self.pressure_grid_val = None
        self.pressure_grid_scoring = None
        self.pressure_grid_test = None

        self.dp_train = None
        self.dp_val = None
        self.dp_scoring = None
        self.dp_test = None

        self.train_regexps = None
        self.train_stride_sample = None
        self.train_filelist = None
        self.val_regexps = None
        self.val_stride_sample = None
        self.val_filelist = None
        self.scoring_regexps = None
        self.scoring_stride_sample = None
        self.scoring_filelist = None
        self.test_regexps = None
        self.test_stride_sample = None
        self.test_filelist = None

        self.full_vars = False
        self.microphysics_constraint = False

        # physical constants from E3SM_ROOT/share/util/shr_const_mod.F90
        self.grav    = 9.80616    # acceleration of gravity ~ m/s^2
        self.cp      = 1.00464e3  # specific heat of dry air   ~ J/kg/K
        self.lv      = 2.501e6    # latent heat of evaporation ~ J/kg
        self.lf      = 3.337e5    # latent heat of fusion      ~ J/kg
        self.lsub    = self.lv + self.lf    # latent heat of sublimation ~ J/kg
        self.rho_air = 101325/(6.02214e26*1.38065e-23/28.966)/273.15 # density of dry air at STP  ~ kg/m^3
                                                                    # ~ 1.2923182846924677
                                                                    # SHR_CONST_PSTD/(SHR_CONST_RDAIR*SHR_CONST_TKFRZ)
                                                                    # SHR_CONST_RDAIR   = SHR_CONST_RGAS/SHR_CONST_MWDAIR
                                                                    # SHR_CONST_RGAS    = SHR_CONST_AVOGAD*SHR_CONST_BOLTZ
        self.rho_h20 = 1.e3       # density of fresh water     ~ kg/m^ 3
        
        self.v1_inputs = ['state_t',
                          'state_q0001',
                          'state_ps',
                          'pbuf_SOLIN',
                          'pbuf_LHFLX',
                          'pbuf_SHFLX']

        self.v1_dyn_inputs = ['state_t',
                              'state_q0001',
                              'state_t_dyn',
                              'state_q0_dyn',
                              'state_ps',
                              'pbuf_SOLIN',
                              'pbuf_LHFLX',
                              'pbuf_SHFLX']
        
        self.v1_outputs = ['ptend_t',
                           'ptend_q0001',
                           'cam_out_NETSW',
                           'cam_out_FLWDS',
                           'cam_out_PRECSC',
                           'cam_out_PRECC',
                           'cam_out_SOLS',
                           'cam_out_SOLL',
                           'cam_out_SOLSD',
                           'cam_out_SOLLD']

        self.v2_inputs = ['state_t',
                          'state_q0001',
                          'state_q0002',
                          'state_q0003',
                          'state_u',
                          'state_v',
                          'state_ps',
                          'pbuf_SOLIN',
                          'pbuf_LHFLX',
                          'pbuf_SHFLX',
                          'pbuf_TAUX',
                          'pbuf_TAUY',
                          'pbuf_COSZRS',
                          'cam_in_ALDIF',
                          'cam_in_ALDIR',
                          'cam_in_ASDIF',
                          'cam_in_ASDIR',
                          'cam_in_LWUP',
                          'cam_in_ICEFRAC',
                          'cam_in_LANDFRAC',
                          'cam_in_OCNFRAC',
                          'cam_in_SNOWHICE',
                          'cam_in_SNOWHLAND',
                          'pbuf_ozone', # outside of the upper troposphere lower stratosphere (UTLS, corresponding to indices 5-21), variance in minimal for these last 3 
                          'pbuf_CH4',
                          'pbuf_N2O'] 

        self.v2_dyn_inputs = ['state_t',
                          'state_q0001',
                          'state_q0002',
                          'state_q0003',
                          'state_u',
                          'state_v',
                          'state_t_dyn',
                          'state_q0_dyn',
                          'state_u_dyn',
                          'state_v_dyn',
                          'state_ps',
                          'pbuf_SOLIN',
                          'pbuf_LHFLX',
                          'pbuf_SHFLX',
                          'pbuf_TAUX',
                          'pbuf_TAUY',
                          'pbuf_COSZRS',
                          'cam_in_ALDIF',
                          'cam_in_ALDIR',
                          'cam_in_ASDIF',
                          'cam_in_ASDIR',
                          'cam_in_LWUP',
                          'cam_in_ICEFRAC',
                          'cam_in_LANDFRAC',
                          'cam_in_OCNFRAC',
                          'cam_in_SNOWHICE',
                          'cam_in_SNOWHLAND',
                          'pbuf_ozone', # outside of the upper troposphere lower stratosphere (UTLS, corresponding to indices 5-21), variance in minimal for these last 3 
                          'pbuf_CH4',
                          'pbuf_N2O'] 

        self.v2_rh_inputs = ['state_t',
                          'state_rh',
                          'state_q0002',
                          'state_q0003',
                          'state_u',
                          'state_v',
                          'pbuf_ozone', # outside of the upper troposphere lower stratosphere (UTLS, corresponding to indices 5-21), variance in minimal for these last 3 
                          'pbuf_CH4',
                          'pbuf_N2O',
                          'state_ps',
                          'pbuf_SOLIN',
                          'pbuf_LHFLX',
                          'pbuf_SHFLX',
                          'pbuf_TAUX',
                          'pbuf_TAUY',
                          'pbuf_COSZRS',
                          'cam_in_ALDIF',
                          'cam_in_ALDIR',
                          'cam_in_ASDIF',
                          'cam_in_ASDIR',
                          'cam_in_LWUP',
                          'cam_in_ICEFRAC',
                          'cam_in_LANDFRAC',
                          'cam_in_OCNFRAC',
                          'cam_in_SNOWHICE',
                          'cam_in_SNOWHLAND'] 
        
        self.v2_rh_mc_inputs = ['state_t',
                          'state_rh',
                          'state_qn',
                          'liq_partition',
                          'state_u',
                          'state_v',
                          'pbuf_ozone', # outside of the upper troposphere lower stratosphere (UTLS, corresponding to indices 5-21), variance in minimal for these last 3 
                          'pbuf_CH4',
                          'pbuf_N2O',
                          'state_ps',
                          'pbuf_SOLIN',
                          'pbuf_LHFLX',
                          'pbuf_SHFLX',
                          'pbuf_TAUX',
                          'pbuf_TAUY',
                          'pbuf_COSZRS',
                          'cam_in_ALDIF',
                          'cam_in_ALDIR',
                          'cam_in_ASDIF',
                          'cam_in_ASDIR',
                          'cam_in_LWUP',
                          'cam_in_ICEFRAC',
                          'cam_in_LANDFRAC',
                          'cam_in_OCNFRAC',
                          'cam_in_SNOWHICE',
                          'cam_in_SNOWHLAND'] 
                
        self.v3_inputs = ['state_t',
                          'state_q0001',
                          'state_q0002',
                          'state_q0003',
                          'state_u',
                          'state_v',
                          'state_t_dyn',
                          'state_q0_dyn',
                          'state_u_dyn',
                          'pbuf_ozone', # outside of the upper troposphere lower stratosphere (UTLS, corresponding to indices 5-21), variance in minimal for these last 3 
                          'pbuf_CH4',
                          'pbuf_N2O',
                          'state_ps',
                          'pbuf_SOLIN',
                          'pbuf_LHFLX',
                          'pbuf_SHFLX',
                          'pbuf_TAUX',
                          'pbuf_TAUY',
                          'pbuf_COSZRS',
                          'cam_in_ALDIF',
                          'cam_in_ALDIR',
                          'cam_in_ASDIF',
                          'cam_in_ASDIR',
                          'cam_in_LWUP',
                          'cam_in_ICEFRAC',
                          'cam_in_LANDFRAC',
                          'cam_in_OCNFRAC',
                          'cam_in_SNOWHICE',
                          'cam_in_SNOWHLAND',] 
    
        self.v4_inputs = ['state_t',
                            'state_rh',
                            'state_q0002',
                            'state_q0003',
                            'state_u',
                            'state_v',
                            'state_t_dyn',
                            'state_q0_dyn',
                            'state_u_dyn',
                            'tm_state_t_dyn',
                            'tm_state_q0_dyn',
                            'tm_state_u_dyn',
                            'state_t_prvphy',
                            'state_q0001_prvphy',
                            'state_q0002_prvphy',
                            'state_q0003_prvphy',
                            'state_u_prvphy',
                            'tm_state_t_prvphy',
                            'tm_state_q0001_prvphy',
                            'tm_state_q0002_prvphy',
                            'tm_state_q0003_prvphy',
                            'tm_state_u_prvphy',
                            'pbuf_ozone', # outside of the upper troposphere lower stratosphere (UTLS, corresponding to indices 5-21), variance in minimal for these last 3 
                            'pbuf_CH4',
                            'pbuf_N2O',
                            'state_ps',
                            # 'pbuf_SOLIN_pm',
                            'pbuf_SOLIN',
                            'pbuf_LHFLX',
                            'pbuf_SHFLX',
                            'pbuf_TAUX',
                            'pbuf_TAUY',
                            # 'pbuf_COSZRS_pm',
                            'pbuf_COSZRS',
                            'cam_in_ALDIF',
                            'cam_in_ALDIR',
                            'cam_in_ASDIF',
                            'cam_in_ASDIR',
                            'cam_in_LWUP',
                            'cam_in_ICEFRAC',
                            'cam_in_LANDFRAC',
                            'cam_in_OCNFRAC',
                            'cam_in_SNOWHICE',
                            'cam_in_SNOWHLAND',
                            'tm_state_ps',
                            'tm_pbuf_SOLIN',
                            'tm_pbuf_LHFLX',
                            'tm_pbuf_SHFLX',
                            'tm_pbuf_COSZRS',
                            'clat',
                            'slat',
                            'icol',] 
        
        self.v5_inputs = ['state_t',
                            'state_rh',
                            'state_qn',
                            'liq_partition',
                            'state_u',
                            'state_v',
                            'state_t_dyn',
                            'state_q0_dyn',
                            'state_u_dyn',
                            'tm_state_t_dyn',
                            'tm_state_q0_dyn',
                            'tm_state_u_dyn',
                            'state_t_prvphy',
                            'state_q0001_prvphy',
                            'state_qn_prvphy',
                            'state_u_prvphy',
                            'tm_state_t_prvphy',
                            'tm_state_q0001_prvphy',
                            'tm_state_qn_prvphy',
                            'tm_state_u_prvphy',
                            'pbuf_ozone', # outside of the upper troposphere lower stratosphere (UTLS, corresponding to indices 5-21), variance in minimal for these last 3 
                            'pbuf_CH4',
                            'pbuf_N2O',
                            'state_ps',
                            # 'pbuf_SOLIN_pm',
                            'pbuf_SOLIN',
                            'pbuf_LHFLX',
                            'pbuf_SHFLX',
                            'pbuf_TAUX',
                            'pbuf_TAUY',
                            # 'pbuf_COSZRS_pm',
                            'pbuf_COSZRS',
                            'cam_in_ALDIF',
                            'cam_in_ALDIR',
                            'cam_in_ASDIF',
                            'cam_in_ASDIR',
                            'cam_in_LWUP',
                            'cam_in_ICEFRAC',
                            'cam_in_LANDFRAC',
                            'cam_in_OCNFRAC',
                            'cam_in_SNOWHICE',
                            'cam_in_SNOWHLAND',
                            'tm_state_ps',
                            'tm_pbuf_SOLIN',
                            'tm_pbuf_LHFLX',
                            'tm_pbuf_SHFLX',
                            'tm_pbuf_COSZRS',
                            'clat',
                            'slat',
                            'icol',]

        self.v6_inputs = ['state_t',
                            'state_rh',
                            'state_qn',
                            'liq_partition',
                            'state_u',
                            'state_v',
                            'state_t_dyn',
                            'state_q0_dyn',
                            'state_u_dyn',
                            'tm_state_t_dyn',
                            'tm_state_q0_dyn',
                            'tm_state_u_dyn',
                            'state_t_prvphy',
                            'state_q0001_prvphy',
                            'state_qn_prvphy',
                            'state_u_prvphy',
                            'tm_state_t_prvphy',
                            'tm_state_q0001_prvphy',
                            'tm_state_qn_prvphy',
                            'tm_state_u_prvphy',
                            'pbuf_ozone',
                            'pbuf_CH4',
                            'pbuf_N2O',
                            'state_ps',
                            'pbuf_SOLIN',
                            'pbuf_LHFLX',
                            'pbuf_SHFLX',
                            'pbuf_TAUX',
                            'pbuf_TAUY',
                            'pbuf_COSZRS',
                            'cam_in_ALDIF',
                            'cam_in_ALDIR',
                            'cam_in_ASDIF',
                            'cam_in_ASDIR',
                            'cam_in_LWUP',
                            'cam_in_ICEFRAC',
                            'cam_in_LANDFRAC',
                            'cam_in_OCNFRAC',
                            'cam_in_SNOWHICE',
                            'cam_in_SNOWHLAND',
                            'clat',
                            'slat',] 
                
        self.v2_outputs = ['ptend_t',
                           'ptend_q0001',
                           'ptend_q0002',
                           'ptend_q0003',
                           'ptend_u',
                           'ptend_v',
                           'cam_out_NETSW',
                           'cam_out_FLWDS',
                           'cam_out_PRECSC',
                           'cam_out_PRECC',
                           'cam_out_SOLS',
                           'cam_out_SOLL',
                           'cam_out_SOLSD',
                           'cam_out_SOLLD']
        
        self.v2_rh_mc_outputs = ['ptend_t',
                           'ptend_q0001',
                           'ptend_qn',
                           'ptend_u',
                           'ptend_v',
                           'cam_out_NETSW',
                           'cam_out_FLWDS',
                           'cam_out_PRECSC',
                           'cam_out_PRECC',
                           'cam_out_SOLS',
                           'cam_out_SOLL',
                           'cam_out_SOLSD',
                           'cam_out_SOLLD']
        
        self.v3_outputs = ['ptend_t',
                           'ptend_q0001',
                           'ptend_q0002',
                           'ptend_q0003',
                           'ptend_u',
                           'ptend_v',
                           'cam_out_NETSW',
                           'cam_out_FLWDS',
                           'cam_out_PRECSC',
                           'cam_out_PRECC',
                           'cam_out_SOLS',
                           'cam_out_SOLL',
                           'cam_out_SOLSD',
                           'cam_out_SOLLD']
        
        self.v4_outputs = ['ptend_t',
                           'ptend_q0001',
                           'ptend_q0002',
                           'ptend_q0003',
                           'ptend_u',
                           'ptend_v',
                           'cam_out_NETSW',
                           'cam_out_FLWDS',
                           'cam_out_PRECSC',
                           'cam_out_PRECC',
                           'cam_out_SOLS',
                           'cam_out_SOLL',
                           'cam_out_SOLSD',
                           'cam_out_SOLLD']
        
        self.v5_outputs = ['ptend_t',
                           'ptend_q0001',
                           'ptend_qn',
                           'ptend_u',
                           'ptend_v',
                           'cam_out_NETSW',
                           'cam_out_FLWDS',
                           'cam_out_PRECSC',
                           'cam_out_PRECC',
                           'cam_out_SOLS',
                           'cam_out_SOLL',
                           'cam_out_SOLSD',
                           'cam_out_SOLLD']

        self.v6_outputs = ['ptend_t',
                           'ptend_q0001',
                           'ptend_qn',
                           'ptend_u',
                           'ptend_v',
                           'cam_out_NETSW',
                           'cam_out_FLWDS',
                           'cam_out_PRECSC',
                           'cam_out_PRECC',
                           'cam_out_SOLS',
                           'cam_out_SOLL',
                           'cam_out_SOLSD',
                           'cam_out_SOLLD',]

        self.var_lens = {#inputs
                        'state_t':self.num_levels,
                        'state_rh':self.num_levels,
                        'state_q0001':self.num_levels,
                        'state_q0002':self.num_levels,
                        'state_q0003':self.num_levels,
                        'state_qn':self.num_levels,
                        'liq_partition':self.num_levels,
                        'state_u':self.num_levels,
                        'state_v':self.num_levels,
                        'state_t_dyn':self.num_levels,
                        'state_q0_dyn':self.num_levels,
                        'state_u_dyn':self.num_levels,
                        'state_v_dyn':self.num_levels,
                        'state_t_prvphy':self.num_levels,
                        'state_q0001_prvphy':self.num_levels,
                        'state_q0002_prvphy':self.num_levels,
                        'state_q0003_prvphy':self.num_levels,
                        'state_qn_prvphy':self.num_levels,
                        'state_u_prvphy':self.num_levels,
                        'tm_state_t_dyn':self.num_levels,
                        'tm_state_q0_dyn':self.num_levels,
                        'tm_state_u_dyn':self.num_levels,
                        'tm_state_t_prvphy':self.num_levels,
                        'tm_state_q0001_prvphy':self.num_levels,
                        'tm_state_q0002_prvphy':self.num_levels,
                        'tm_state_q0003_prvphy':self.num_levels,
                        'tm_state_qn_prvphy':self.num_levels,
                        'tm_state_u_prvphy':self.num_levels,
                        'state_ps':1,
                        'pbuf_SOLIN':1,
                        'pbuf_LHFLX':1,
                        'pbuf_SHFLX':1,
                        'pbuf_TAUX':1,
                        'pbuf_TAUY':1,
                        'pbuf_COSZRS':1,
                        'tm_state_ps':1,
                        'tm_pbuf_SOLIN':1,
                        'tm_pbuf_LHFLX':1,
                        'tm_pbuf_SHFLX':1,
                        'tm_pbuf_COSZRS':1,
                        'cam_in_ALDIF':1,
                        'cam_in_ALDIR':1,
                        'cam_in_ASDIF':1,
                        'cam_in_ASDIR':1,
                        'cam_in_LWUP':1,
                        'cam_in_ICEFRAC':1,
                        'cam_in_LANDFRAC':1,
                        'cam_in_OCNFRAC':1,
                        'cam_in_SNOWHICE':1,
                        'cam_in_SNOWHLAND':1,
                        'pbuf_ozone':self.num_levels,
                        'pbuf_CH4':self.num_levels,
                        'pbuf_N2O':self.num_levels,
                        'clat':1,
                        'slat':1,
                        'icol':1,
                        #outputs
                        'ptend_t':self.num_levels,
                        'ptend_q0001':self.num_levels,
                        'ptend_q0002':self.num_levels,
                        'ptend_q0003':self.num_levels,
                        'ptend_qn':self.num_levels,
                        'ptend_u':self.num_levels,
                        'ptend_v':self.num_levels,
                        'cam_out_NETSW':1,
                        'cam_out_FLWDS':1,
                        'cam_out_PRECSC':1,
                        'cam_out_PRECC':1,
                        'cam_out_SOLS':1,
                        'cam_out_SOLL':1,
                        'cam_out_SOLSD':1,
                        'cam_out_SOLLD':1,
                        'pbuf_SOLIN_pm':1,
                        'pbuf_COSZRS_pm':1,
                        }

        self.var_short_names = {'ptend_t':'$dT/dt$',
                                'ptend_q0001':'$dq/dt$',
                                'cam_out_NETSW':'NETSW',
                                'cam_out_FLWDS':'FLWDS',
                                'cam_out_PRECSC':'PRECSC',
                                'cam_out_PRECC':'PRECC',
                                'cam_out_SOLS':'SOLS',
                                'cam_out_SOLL':'SOLL',
                                'cam_out_SOLSD':'SOLSD',
                                'cam_out_SOLLD':'SOLLD'}
        
        self.target_energy_conv = {'ptend_t':self.cp,
                                   'ptend_q0001':self.lv,
                                   'ptend_q0002':self.lv,
                                   'ptend_q0003':self.lv,
                                   'ptend_qn':self.lv,
                                   'ptend_wind': None,
                                   'cam_out_NETSW':1.,
                                   'cam_out_FLWDS':1.,
                                   'cam_out_PRECSC':self.lv*self.rho_h20,
                                   'cam_out_PRECC':self.lv*self.rho_h20,
                                   'cam_out_SOLS':1.,
                                   'cam_out_SOLL':1.,
                                   'cam_out_SOLSD':1.,
                                   'cam_out_SOLLD':1.
                                  }

        # for metrics
    
        self.input_train = None
        self.target_train = None
        self.preds_train = None
        self.samplepreds_train = None
        self.target_weighted_train = {}
        self.preds_weighted_train = {}
        self.samplepreds_weighted_train = {}
        self.metrics_train = []
        self.metrics_idx_train = {}
        self.metrics_var_train = {}

        self.input_val = None
        self.target_val = None
        self.preds_val = None
        self.samplepreds_val = None
        self.target_weighted_val = {}
        self.preds_weighted_val = {}
        self.samplepreds_weighted_val = {}
        self.metrics_val = []
        self.metrics_idx_val = {}
        self.metrics_var_val = {}
        
        self.input_scoring = None
        self.target_scoring = None
        self.preds_scoring = None
        self.samplepreds_scoring = None
        self.target_weighted_scoring = {}
        self.preds_weighted_scoring = {}
        self.samplepreds_weighted_scoring = {}
        self.metrics_scoring = []
        self.metrics_idx_scoring = {}
        self.metrics_var_scoring = {}

        self.input_test = None
        self.target_test = None
        self.preds_test = None
        self.samplepreds_test = None
        self.target_weighted_test = {}
        self.preds_weighted_test = {}
        self.samplepreds_weighted_test = {}
        self.metrics_test = []
        self.metrics_idx_test = {}
        self.metrics_var_test = {}

        self.model_names = []
        self.metrics_names = []
        self.metrics_dict = {'MAE': self.calc_MAE,
                             'RMSE': self.calc_RMSE,
                             'R2': self.calc_R2,
                             'CRPS': self.calc_CRPS,
                             'bias': self.calc_bias
                            }
        self.num_CRPS = 32
        self.linecolors = ['#0072B2', 
                           '#E69F00', 
                           '#882255', 
                           '#009E73', 
                           '#D55E00'
                           ]

    def set_to_v1_vars(self):
        '''
        This function sets the inputs and outputs to the V1 subset.
        It also indicates the index of the surface pressure variable.
        '''
        self.input_vars = self.v1_inputs
        self.target_vars = self.v1_outputs
        self.ps_index = 120
        self.full_vars = False
        self.input_profile_num = sum(1 for input_var in self.input_vars if self.var_lens[input_var] == 60)
        self.input_scalar_num = sum(1 for input_var in self.input_vars if self.var_lens[input_var] == 1)
        self.target_profile_num = sum(1 for target_var in self.target_vars if self.var_lens[target_var] == 60)
        self.target_scalar_num = sum(1 for target_var in self.target_vars if self.var_lens[target_var] == 1)
        self.input_feature_len = self.input_profile_num * 60 + self.input_scalar_num # 124
        self.target_feature_len = self.target_profile_num * 60 + self.target_scalar_num # 128

    def set_to_v1_dyn_vars(self):
        '''
        This function sets the inputs and outputs to the V1_dyn subset.
        It also indicates the index of the surface pressure variable.
        '''
        self.input_vars = self.v1_dyn_inputs
        self.target_vars = self.v1_outputs
        self.ps_index = 240
        self.full_vars = False
        self.input_profile_num = sum(1 for input_var in self.input_vars if self.var_lens[input_var] == 60)
        self.input_scalar_num = sum(1 for input_var in self.input_vars if self.var_lens[input_var] == 1)
        self.target_profile_num = sum(1 for target_var in self.target_vars if self.var_lens[target_var] == 60)
        self.target_scalar_num = sum(1 for target_var in self.target_vars if self.var_lens[target_var] == 1)
        self.input_feature_len = self.input_profile_num * 60 + self.input_scalar_num # 224
        self.target_feature_len = self.target_profile_num * 60 + self.target_scalar_num # 128

    def set_to_v2_vars(self):
        '''
        This function sets the inputs and outputs to the V2 subset.
        It also indicates the index of the surface pressure variable.
        '''
        self.input_vars = self.v2_inputs
        self.target_vars = self.v2_outputs
        self.ps_index = 360
        self.full_vars = True
        self.input_profile_num = sum(1 for input_var in self.input_vars if self.var_lens[input_var] == 60)
        self.input_scalar_num = sum(1 for input_var in self.input_vars if self.var_lens[input_var] == 1)
        self.target_profile_num = sum(1 for target_var in self.target_vars if self.var_lens[target_var] == 60)
        self.target_scalar_num = sum(1 for target_var in self.target_vars if self.var_lens[target_var] == 1)
        self.input_feature_len = self.input_profile_num * 60 + self.input_scalar_num # 557
        self.target_feature_len = self.target_profile_num * 60 + self.target_scalar_num # 368

    def set_to_v2_rh_vars(self):
        '''
        This function sets the inputs and outputs to the V2 subset.
        It also indicates the index of the surface pressure variable.
        '''
        self.input_vars = self.v2_rh_inputs
        self.target_vars = self.v2_outputs
        self.ps_index = 360
        self.full_vars = True
        self.input_profile_num = sum(1 for input_var in self.input_vars if self.var_lens[input_var] == 60)
        self.input_scalar_num = sum(1 for input_var in self.input_vars if self.var_lens[input_var] == 1)
        self.target_profile_num = sum(1 for target_var in self.target_vars if self.var_lens[target_var] == 60)
        self.target_scalar_num = sum(1 for target_var in self.target_vars if self.var_lens[target_var] == 1)
        self.input_feature_len = self.input_profile_num * 60 + self.input_scalar_num # 557
        self.target_feature_len = self.target_profile_num * 60 + self.target_scalar_num # 368

    def set_to_v2_rh_mc_vars(self):
        '''
        This function sets the inputs and outputs to the V2 subset.
        It also indicates the index of the surface pressure variable.
        '''
        self.input_vars = self.v2_rh_mc_inputs
        self.target_vars = self.v2_rh_mc_outputs
        self.ps_index = 360
        self.full_vars = False
        self.microphysics_constraint = True
        self.input_profile_num = sum(1 for input_var in self.input_vars if self.var_lens[input_var] == 60)
        self.input_scalar_num = sum(1 for input_var in self.input_vars if self.var_lens[input_var] == 1)
        self.target_profile_num = sum(1 for target_var in self.target_vars if self.var_lens[target_var] == 60)
        self.target_scalar_num = sum(1 for target_var in self.target_vars if self.var_lens[target_var] == 1)
        self.input_feature_len = self.input_profile_num * 60 + self.input_scalar_num # 557
        self.target_feature_len = self.target_profile_num * 60 + self.target_scalar_num # 308

    def set_to_v2_dyn_vars(self):
        '''
        This function sets the inputs and outputs to the V2_dyn subset.
        It also indicates the index of the surface pressure variable.
        '''
        self.input_vars = self.v2_dyn_inputs
        self.target_vars = self.v2_outputs
        self.ps_index = 600
        self.full_vars = True
        self.input_profile_num = sum(1 for input_var in self.input_vars if self.var_lens[input_var] == 60)
        self.input_scalar_num = sum(1 for input_var in self.input_vars if self.var_lens[input_var] == 1)
        self.target_profile_num = sum(1 for target_var in self.target_vars if self.var_lens[target_var] == 60)
        self.target_scalar_num = sum(1 for target_var in self.target_vars if self.var_lens[target_var] == 1)
        self.input_feature_len = self.input_profile_num * 60 + self.input_scalar_num # 797
        self.target_feature_len = self.target_profile_num * 60 + self.target_scalar_num # 368

    def set_to_v3_vars(self):
        '''
        This function sets the inputs and outputs to the V3 subset.
        It also indicates the index of the surface pressure variable.
        '''
        self.input_vars = self.v3_inputs
        self.target_vars = self.v3_outputs
        self.ps_index = 720
        self.full_vars = True
        self.input_profile_num = sum(1 for input_var in self.input_vars if self.var_lens[input_var] == 60)
        self.input_scalar_num = sum(1 for input_var in self.input_vars if self.var_lens[input_var] == 1)
        self.target_profile_num = sum(1 for target_var in self.target_vars if self.var_lens[target_var] == 60)
        self.target_scalar_num = sum(1 for target_var in self.target_vars if self.var_lens[target_var] == 1)
        self.input_feature_len = self.input_profile_num * 60 + self.input_scalar_num # 737
        self.target_feature_len = self.target_profile_num * 60 + self.target_scalar_num # 368

    def set_to_v4_vars(self):
        '''
        This function sets the inputs and outputs to the V3 subset.
        It also indicates the index of the surface pressure variable.
        '''
        self.input_vars = self.v4_inputs
        self.target_vars = self.v4_outputs
        self.ps_index = 1500
        self.full_vars = True
        self.input_profile_num = sum(1 for input_var in self.input_vars if self.var_lens[input_var] == 60)
        self.input_scalar_num = sum(1 for input_var in self.input_vars if self.var_lens[input_var] == 1)
        self.target_profile_num = sum(1 for target_var in self.target_vars if self.var_lens[target_var] == 60)
        self.target_scalar_num = sum(1 for target_var in self.target_vars if self.var_lens[target_var] == 1)
        self.input_feature_len = self.input_profile_num * 60 + self.input_scalar_num # 1525
        self.target_feature_len = self.target_profile_num * 60 + self.target_scalar_num # 368
    
    def set_to_v5_vars(self):
        '''
        This function sets the inputs and outputs to the V3 subset.
        It also indicates the index of the surface pressure variable.
        '''
        self.input_vars = self.v5_inputs
        self.target_vars = self.v5_outputs
        self.ps_index = 1380
        self.full_vars = False
        self.microphysics_constraint = True
        self.input_profile_num = sum(1 for input_var in self.input_vars if self.var_lens[input_var] == 60)
        self.input_scalar_num = sum(1 for input_var in self.input_vars if self.var_lens[input_var] == 1)
        self.target_profile_num = sum(1 for target_var in self.target_vars if self.var_lens[target_var] == 60)
        self.target_scalar_num = sum(1 for target_var in self.target_vars if self.var_lens[target_var] == 1)
        self.input_feature_len = self.input_profile_num * 60 + self.input_scalar_num # 1405 = 737 + 788 - 120
        self.target_feature_len = self.target_profile_num * 60 + self.target_scalar_num # 368

    def set_to_v6_vars(self):
        '''
        This function sets the inputs and outputs to the V6 subset.
        It also indicates the index of the surface pressure variable.
        '''
        self.input_vars = self.v6_inputs
        self.target_vars = self.v6_outputs
        self.ps_index = 1380
        self.full_vars = False
        self.microphysics_constraint = True
        self.input_profile_num = sum(1 for input_var in self.input_vars if self.var_lens[input_var] == 60)
        self.input_scalar_num = sum(1 for input_var in self.input_vars if self.var_lens[input_var] == 1)
        self.target_profile_num = sum(1 for target_var in self.target_vars if self.var_lens[target_var] == 60)
        self.target_scalar_num = sum(1 for target_var in self.target_vars if self.var_lens[target_var] == 1)
        self.input_feature_len = self.input_profile_num * 60 + self.input_scalar_num # 1399
        self.target_feature_len = self.target_profile_num * 60 + self.target_scalar_num # 308

    def eliq(self, T):
        """
        Function taking temperature (in K) and outputting liquid saturation
        pressure (in hPa) using a polynomial fit
        """
        a_liq = np.array([-0.976195544e-15,-0.952447341e-13,0.640689451e-10,
                                0.206739458e-7,0.302950461e-5,0.264847430e-3,
                                0.142986287e-1,0.443987641,6.11239921]);
        c_liq = -80
        T0 = 273.16
        return 100*np.polyval(a_liq,np.maximum(c_liq,T-T0))

    def eice(self, T):
        """
        Function taking temperature (in K) and outputting ice saturation
        pressure (in hPa) using a polynomial fit
        """
        a_ice = np.array([0.252751365e-14,0.146898966e-11,0.385852041e-9,
                        0.602588177e-7,0.615021634e-5,0.420895665e-3,
                        0.188439774e-1,0.503160820,6.11147274]);
        c_ice = np.array([273.15,185,-100,0.00763685,0.000151069,7.48215e-07])
        T0 = 273.16
        return (T>c_ice[0])*self.eliq(T)+\
        (T<=c_ice[0])*(T>c_ice[1])*100*np.polyval(a_ice,T-T0)+\
        (T<=c_ice[1])*100*(c_ice[3]+np.maximum(c_ice[2],T-T0)*\
                        (c_ice[4]+np.maximum(c_ice[2],T-T0)*c_ice[5]))

    def get_xrdata(self, file, file_vars = None):
        '''
        This function reads in a file and returns an xarray dataset with the variables specified.
        file_vars must be a list of strings.
        '''
        ds = xr.open_dataset(file, engine = 'netcdf4')
        if file_vars is not None:
            # if "state_rh" is in file_vars but not in ds, then add it to ds
            if 'state_rh' in file_vars and 'state_rh' not in ds:
                tair = ds['state_t']
                T0 = 273.16 # Freezing temperature in standard conditions
                T00 = 253.16 # Temperature below which we use e_ice
                omega = (tair - T00) / (T0 - T00)
                omega = np.maximum( 0, np.minimum( 1, omega ))
                esat =  omega * self.eliq(tair) + (1-omega) * self.eice(tair)
                Rd = 287 # Specific gas constant for dry air
                Rv = 461 # Specific gas constant for water vapor 
                if 'state_pmid' not in ds:
                    print(file) 
                qvs = (Rd*esat)/(Rv*ds['state_pmid'])
                state_rh = ds['state_q0001']/qvs
                ds['state_rh'] = state_rh

            # if "icol" is in file_vars but not in ds, then add it to ds
            if 'icol' in file_vars and 'icol' not in ds:
                lat = ds['lat']
                icol = lat.copy()
                icol[:] = np.arange(1,385)
                ds['icol'] = icol
            
            if 'liq_partition' in file_vars and 'liq_partition' not in ds:
                tair = ds['state_t']
                T0 = 273.16 # Freezing temperature in standard conditions
                T00 = 253.16 # Temperature below which we use e_ice
                liq_partition = (tair - T00) / (T0 - T00)
                liq_partition = np.maximum( 0, np.minimum( 1, liq_partition ))
                ds['liq_partition'] = liq_partition
            
            if 'state_qn' in file_vars and 'state_qn' not in ds:
                state_qn = ds['state_q0002'] + ds['state_q0003']
                ds['state_qn'] = state_qn

            if 'state_qn_prvphy' in file_vars and 'state_qn_prvphy' not in ds:
                state_qn_prvphy = ds['state_q0002_prvphy'] + ds['state_q0003_prvphy']
                ds['state_qn_prvphy'] = state_qn_prvphy

            if 'tm_state_qn_prvphy' in file_vars and 'tm_state_qn_prvphy' not in ds:
                tm_state_qn_prvphy = ds['tm_state_q0002_prvphy'] + ds['tm_state_q0003_prvphy']
                ds['tm_state_qn_prvphy'] = tm_state_qn_prvphy 

        if file_vars is not None:
            ds = ds[file_vars]
        ds = ds.merge(self.grid_info[['lat','lon']], compat='override')
        ds = ds.where((ds['lat']>-999)*(ds['lat']<999), drop=True)
        ds = ds.where((ds['lon']>-999)*(ds['lon']<999), drop=True)
        return ds

    def get_input(self, input_file):
        '''
        This function reads in a file and returns an xarray dataset with the input variables for the emulator.
        '''
        # read inputs
        return self.get_xrdata(input_file, self.input_vars)

    def get_target(self, input_file):
        '''
        This function reads in a file and returns an xarray dataset with the target variables for the emulator.
        '''
        tmp_input_vars = self.input_vars
        if 'state_q0001' not in input_file: 
            tmp_input_vars = tmp_input_vars + ['state_q0001']
        if 'state_q0002' not in input_file:
            tmp_input_vars = tmp_input_vars + ['state_q0002']
        if 'state_q0003' not in input_file:
            tmp_input_vars = tmp_input_vars + ['state_q0003']
        
        ds_input = self.get_xrdata(input_file, tmp_input_vars)
        # # read inputs
        # if 'state_q0001' not in input_file: 
        #     ds_input = self.get_xrdata(input_file, self.input_vars+['state_q0001'])
        # else:
        #     ds_input = self.get_input(input_file)
        print("input file")
        print(input_file)
        ds_target = self.get_xrdata(input_file.replace(f'.{self.input_abbrev}.',f'.{self.output_abbrev}.'))
        print(input_file.replace(f'.{self.input_abbrev}.',f'.{self.output_abbrev}.'))
        # each timestep is 20 minutes which corresponds to 1200 seconds
        ds_target['ptend_t'] = (ds_target['state_t'] - ds_input['state_t'])/1200 # T tendency [K/s]
        ds_target['ptend_q0001'] = (ds_target['state_q0001'] - ds_input['state_q0001'])/1200 # Q tendency [kg/kg/s]
        if self.full_vars:
            ds_target['ptend_q0002'] = (ds_target['state_q0002'] - ds_input['state_q0002'])/1200 # Q tendency [kg/kg/s]
            ds_target['ptend_q0003'] = (ds_target['state_q0003'] - ds_input['state_q0003'])/1200 # Q tendency [kg/kg/s]
            ds_target['ptend_u'] = (ds_target['state_u'] - ds_input['state_u'])/1200 # U tendency [m/s/s]
            ds_target['ptend_v'] = (ds_target['state_v'] - ds_input['state_v'])/1200 # V tendency [m/s/s]   
        elif self.microphysics_constraint:
            ds_target['ptend_qn'] = (ds_target['state_q0002'] - ds_input['state_q0002'] + ds_target['state_q0003'] - ds_input['state_q0003'])/1200
            ds_target['ptend_u'] = (ds_target['state_u'] - ds_input['state_u'])/1200 # U tendency [m/s/s]
            ds_target['ptend_v'] = (ds_target['state_v'] - ds_input['state_v'])/1200
        ds_target = ds_target[self.target_vars]
        return ds_target
    
    def set_regexps(self, data_split: str, regexps: List[str]) -> None:
        '''
        This function sets the regular expressions used for getting the filelist for train, val, scoring, and test.
        '''
        assert data_split in ['train', 'val', 'scoring', 'test'], 'Provided data_split is not valid. Available options are train, val, scoring, and test.'
        assert isinstance(regexps, list), 'Provided regexps is not a list.'
        if data_split == 'train':
            self.train_regexps = regexps
        elif data_split == 'val':
            self.val_regexps = regexps
        elif data_split == 'scoring':
            self.scoring_regexps = regexps
        elif data_split == 'test':
            self.test_regexps = regexps
    
    def set_stride_sample(self, data_split, stride_sample):
        '''
        This function sets the stride_sample for train, val, scoring, and test.
        '''
        assert data_split in ['train', 'val', 'scoring', 'test'], 'Provided data_split is not valid. Available options are train, val, scoring, and test.'
        if data_split == 'train':
            self.train_stride_sample = stride_sample
        elif data_split == 'val':
            self.val_stride_sample = stride_sample
        elif data_split == 'scoring':
            self.scoring_stride_sample = stride_sample
        elif data_split == 'test':
            self.test_stride_sample = stride_sample
    
    def set_filelist(self, data_split, start_idx = 0, end_idx = -1, num_resubmit = None, num_resubmit_abandon=None, period=None, samples_to_keep=None):
        '''
        This function sets the filelists corresponding to data splits for train, val, scoring, and test.
        '''
        filelist = []
        assert data_split in ['train', 'val', 'scoring', 'test'], 'Provided data_split is not valid. Available options are train, val, scoring, and test.'

        # if num_resubmit is not None and num_resubmit_abandon is not None, then assert that stride_sample needs to be 1

        if data_split == 'train':
            assert self.train_regexps is not None, 'regexps for train is not set.'
            assert self.train_stride_sample is not None, 'stride_sample for train is not set.'
            # if num_resubmit is not None and num_resubmit_abandon is not None:
            #     assert self.train_stride_sample == 1, 'stride_sample for train needs to be 1.'
            for regexp in self.train_regexps:
                filelist = filelist + glob.glob(self.data_path + "*/" + regexp)

            sorted_list = sorted(filelist)
            if num_resubmit is not None and num_resubmit_abandon is not None:
                sorted_list = [file for i, file in enumerate(sorted_list) if i%num_resubmit >= num_resubmit_abandon]
            self.train_filelist = sorted_list[start_idx:end_idx:self.train_stride_sample]
            if (period is not None) and (samples_to_keep is not None):
                self.train_filelist = [file for i, file in enumerate(self.train_filelist) if i%period < samples_to_keep]
            # self.train_filelist = sorted(filelist)[start_idx:end_idx:self.train_stride_sample] # -1 to avoid the last file in mlis does not have corresponding mlos due to model crash
            # # for every num_resubmit elements in filelist, abandon the first num_resubmit_abandon elements
            # if num_resubmit is not None and num_resubmit_abandon is not None:
            #     self.train_filelist = [file for i, file in enumerate(self.train_filelist) if i%num_resubmit >= num_resubmit_abandon]

        elif data_split == 'val':
            assert self.val_regexps is not None, 'regexps for val is not set.'
            assert self.val_stride_sample is not None, 'stride_sample for val is not set.'
            for regexp in self.val_regexps:
                filelist = filelist + glob.glob(self.data_path + "*/" + regexp)
            # self.val_filelist = sorted(filelist)[start_idx:end_idx:self.val_stride_sample]
            # if num_resubmit is not None and num_resubmit_abandon is not None:
            #     self.val_filelist = [file for i, file in enumerate(self.val_filelist) if i%num_resubmit >= num_resubmit_abandon]
            sorted_list = sorted(filelist)
            if num_resubmit is not None and num_resubmit_abandon is not None:
                sorted_list = [file for i, file in enumerate(sorted_list) if i%num_resubmit >= num_resubmit_abandon]
            self.val_filelist = sorted_list[start_idx:end_idx:self.val_stride_sample]
            if (period is not None) and (samples_to_keep is not None):
                self.val_filelist = [file for i, file in enumerate(self.val_filelist) if i%period < samples_to_keep]

        elif data_split == 'scoring':
            assert self.scoring_regexps is not None, 'regexps for scoring is not set.'
            assert self.scoring_stride_sample is not None, 'stride_sample for scoring is not set.'
            for regexp in self.scoring_regexps:
                filelist = filelist + glob.glob(self.data_path + "*/" + regexp)
            # self.scoring_filelist = sorted(filelist)[start_idx:end_idx:self.scoring_stride_sample]
            # if num_resubmit is not None and num_resubmit_abandon is not None:
            #     self.scoring_filelist = [file for i, file in enumerate(self.scoring_filelist) if i%num_resubmit >= num_resubmit_abandon]
            sorted_list = sorted(filelist)
            if num_resubmit is not None and num_resubmit_abandon is not None:
                sorted_list = [file for i, file in enumerate(sorted_list) if i%num_resubmit >= num_resubmit_abandon]
            self.scoring_filelist = sorted_list[start_idx:end_idx:self.scoring_stride_sample]
            if (period is not None) and (samples_to_keep is not None):
                self.scoring_filelist = [file for i, file in enumerate(self.scoring_filelist) if i%period < samples_to_keep]
        elif data_split == 'test':
            assert self.test_regexps is not None, 'regexps for test is not set.'
            assert self.test_stride_sample is not None, 'stride_sample for test is not set.'
            for regexp in self.test_regexps:
                filelist = filelist + glob.glob(self.data_path + "*/" + regexp)
            # self.test_filelist = sorted(filelist)[start_idx:end_idx:self.test_stride_sample]
            # if num_resubmit is not None and num_resubmit_abandon is not None:
            #     self.test_filelist = [file for i, file in enumerate(self.test_filelist) if i%num_resubmit >= num_resubmit_abandon]
            sorted_list = sorted(filelist)
            if num_resubmit is not None and num_resubmit_abandon is not None:
                sorted_list = [file for i, file in enumerate(sorted_list) if i%num_resubmit >= num_resubmit_abandon]
            self.test_filelist = sorted_list[start_idx:end_idx:self.test_stride_sample]
            if (period is not None) and (samples_to_keep is not None):
                self.test_filelist = [file for i, file in enumerate(self.test_filelist) if i%period < samples_to_keep]

    def get_filelist(self, data_split):
        '''
        This function returns the filelist corresponding to data splits for train, val, scoring, and test.
        '''
        assert data_split in ['train', 'val', 'scoring', 'test'], 'Provided data_split is not valid. Available options are train, val, scoring, and test.'
        if data_split == 'train':
            assert self.train_filelist is not None, 'filelist for train is not set.'
            return self.train_filelist
        elif data_split == 'val':
            assert self.val_filelist is not None, 'filelist for val is not set.'
            return self.val_filelist
        elif data_split == 'scoring':
            assert self.scoring_filelist is not None, 'filelist for scoring is not set.'
            return self.scoring_filelist
        elif data_split == 'test':
            assert self.test_filelist is not None, 'filelist for test is not set.'
            return self.test_filelist
    
    def load_ncdata_with_generator(self, data_split):
        '''
        This function works as a dataloader when training the emulator with raw netCDF files.
        This can be used as a dataloader during training or it can be used to create entire datasets.
        When used as a dataloader for training, I/O can slow down training considerably.
        This function also normalizes the data.
        mlis corresponds to input
        mlos corresponds to target
        '''
        filelist = self.get_filelist(data_split)
        def gen():
            for file in filelist:
                # read inputs
                ds_input = self.get_input(file)
                # read targets
                ds_target = self.get_target(file)

                if self.qinput_log:
                    if 'state_q0002' in ds_input:
                        ds_input['state_q0002'][:] = np.log10(1+ds_input['state_q0002']*1e6)
                    if 'state_q0003' in ds_input:
                        ds_input['state_q0003'][:] = np.log10(1+ds_input['state_q0003']*1e6)
                
                # normalization, scaling
                if self.normalize:
                    ds_input = (ds_input - self.input_mean)/(self.input_max - self.input_min)
                    ds_target = ds_target*self.output_scale
                else:
                    ds_input = ds_input.drop(['lat','lon'])

                # stack
                # ds = ds.stack({'batch':{'sample','ncol'}})
                ds_input = ds_input.stack({'batch':{'ncol'}})
                ds_input = ds_input.to_stacked_array('mlvar', sample_dims=['batch'], name=self.input_abbrev)
                # dso = dso.stack({'batch':{'sample','ncol'}})
                ds_target = ds_target.stack({'batch':{'ncol'}})
                ds_target = ds_target.to_stacked_array('mlvar', sample_dims=['batch'], name=self.output_abbrev)
                yield (ds_input.values, ds_target.values)
        return gen()
    
    def save_as_npy(self,
                 data_split, 
                 save_path = '',
                 save_latlontime_dict = False):
        '''
        This function saves the training data as a .npy file.
        '''
        data_loader = self.load_ncdata_with_generator(data_split)
        data_list = list(data_loader)
        npy_input = np.concatenate([data_list[x][0] for x in range(len(data_list))])

        if self.normalize:
            npy_input[np.isinf(npy_input)] = 0 # replace inf with 0, some level have constant gas concentration.
            #also replace nans with 0 (e.g., cld in stratosphere, std=0)
            npy_input[np.isnan(npy_input)] = 0

        # if save_path not exist, create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # add "/" to the end of save_path if it does not exist
        if save_path[-1] != '/':
            save_path = save_path + '/'

        npy_input = np.float32(npy_input)
        if self.save_h5:
            h5_path = save_path + data_split + '_input.h5'
            with h5py.File(h5_path, 'w') as hdf:
                # Create a dataset within the HDF5 file
                hdf.create_dataset('data', data=npy_input, dtype=npy_input.dtype)

        if self.save_npy:
            with open(save_path + data_split + '_input.npy', 'wb') as f:
                np.save(f, npy_input)
        
        if self.save_zarr:
            zarr_path = save_path + data_split + '_input.zarr'
            z = zarr.open(zarr_path, mode='w', shape=npy_input.shape, dtype=npy_input.dtype, chunks=(100, npy_input.shape[1]))
            z[:] = npy_input

        del npy_input
        
        npy_target = np.concatenate([data_list[x][1] for x in range(len(data_list))])
        npy_target = np.float32(npy_target)

        if self.save_h5:
            h5_path = save_path + data_split + '_target.h5'
            with h5py.File(h5_path, 'w') as hdf:
                # Create a dataset within the HDF5 file
                hdf.create_dataset('data', data=npy_target, dtype=npy_target.dtype)

        if self.save_npy:
            with open(save_path + data_split + '_target.npy', 'wb') as f:
                np.save(f, npy_target)

        if self.save_zarr:
            zarr_path = save_path + data_split + '_target.zarr'
            z = zarr.open(zarr_path, mode='w', shape=npy_target.shape, dtype=npy_target.dtype, chunks=(100, npy_target.shape[1]))
            z[:] = npy_target

        if data_split == 'train':
            data_files = self.train_filelist
        elif data_split == 'val':
            data_files = self.val_filelist
        elif data_split == 'scoring':
            data_files = self.scoring_filelist
        elif data_split == 'test':
            data_files = self.test_filelist
        if save_latlontime_dict:
            dates = [re.sub(f'^.*{self.input_abbrev}\.', '', x) for x in data_files]
            dates = [re.sub('\.nc$', '', x) for x in dates]
            repeat_dates = []
            for date in dates:
                for i in range(self.num_latlon):
                    repeat_dates.append(date)
            latlontime = {i: [(self.grid_info['lat'].values[i%self.num_latlon], self.grid_info['lon'].values[i%self.num_latlon]), repeat_dates[i]] for i in range(npy_input.shape[0])}
            with open(save_path + data_split + '_indextolatlontime.pkl', 'wb') as f:
                pickle.dump(latlontime, f)

    def load_ncdata_with_generator_inputonly(self, data_split):
        '''
        This function works as a dataloader when training the emulator with raw netCDF files.
        mlis corresponds to input
        '''
        filelist = self.get_filelist(data_split)
        def gen():
            for file in filelist:
                # read inputs
                ds_input = self.get_input(file)
                # read targets
                #ds_target = self.get_target(file)
                
                # normalization, scaling
                if self.normalize:
                    ds_input = (ds_input - self.input_mean)/(self.input_max - self.input_min)
                    #ds_target = ds_target*self.output_scale
                else:
                    ds_input = ds_input.drop(['lat','lon'])

                # stack
                # ds = ds.stack({'batch':{'sample','ncol'}})
                ds_input = ds_input.stack({'batch':{'ncol'}})
                ds_input = ds_input.to_stacked_array('mlvar', sample_dims=['batch'], name=self.input_abbrev)
                # dso = dso.stack({'batch':{'sample','ncol'}})
                #ds_target = ds_target.stack({'batch':{'ncol'}})
                #ds_target = ds_target.to_stacked_array('mlvar', sample_dims=['batch'], name='mlos')
                yield ds_input.values #ds_target.values)
        return gen()

    def save_as_npy_inputonly(self,
                 data_split, 
                 save_path = '',
                 save_latlontime_dict = False):
        '''
        This function saves the training data as a .npy file.
        '''
        data_loader = self.load_ncdata_with_generator_inputonly(data_split)
        data_list = list(data_loader)
        npy_input = np.concatenate([data_list[x][0] for x in range(len(data_list))])
        
        # if save_path not exist, create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # add "/" to the end of save_path if it does not exist
        if save_path[-1] != '/':
            save_path = save_path + '/'

        with open(save_path + data_split + '_input.npy', 'wb') as f:
            np.save(f, np.float32(npy_input))
        # with open(save_path + data_split + '_target.npy', 'wb') as f:
        #     np.save(f, np.float32(npy_target))
        if data_split == 'train':
            data_files = self.train_filelist
        elif data_split == 'val':
            data_files = self.val_filelist
        elif data_split == 'scoring':
            data_files = self.scoring_filelist
        elif data_split == 'test':
            data_files = self.test_filelist
        if save_latlontime_dict:
            dates = [re.sub(f'^.*{self.input_abbrev}\.', '', x) for x in data_files]
            dates = [re.sub('\.nc$', '', x) for x in dates]
            repeat_dates = []
            for date in dates:
                for i in range(self.num_latlon):
                    repeat_dates.append(date)
            latlontime = {i: [(self.grid_info['lat'].values[i%self.num_latlon], self.grid_info['lon'].values[i%self.num_latlon]), repeat_dates[i]] for i in range(npy_input.shape[0])}
            with open(save_path + data_split + '_indextolatlontime.pkl', 'wb') as f:
                pickle.dump(latlontime, f)
    
    def reshape_npy(self, var_arr, var_arr_dim):
        '''
        This function reshapes the a variable in numpy such that time gets its own axis (instead of being num_samples x num_levels).
        Shape of target would be (timestep, lat/lon combo, num_levels)
        '''
        var_arr = var_arr.reshape((int(var_arr.shape[0]/self.num_latlon), self.num_latlon, var_arr_dim))
        return var_arr
    
    def save_norm(self, save_path = '', write=False):
        # calculate norms for input first
        input_sub  = []
        input_div  = []
        fmt = '%.6e'
        for var in self.input_vars:
            var_lev = self.var_lens[var]
            if var_lev == 1:
                input_sub.append(self.input_mean[var].values)
                input_div.append(self.input_max[var].values - self.input_min[var].values)
            else:
                for i in range(var_lev):
                    input_sub.append(self.input_mean[var].values[i])
                    input_div.append(self.input_max[var].values[i] - self.input_min[var].values[i])
        input_sub = np.array(input_sub)
        input_div = np.array(input_div)
        if write:
            np.savetxt(save_path + '/inp_sub.txt', input_sub.reshape(1, -1), fmt=fmt, delimiter=',')
            np.savetxt(save_path + '/inp_div.txt', input_div.reshape(1, -1), fmt=fmt, delimiter=',')
        # calculate norms for target
        out_scale = []
        for var in self.target_vars:
            var_lev = self.var_lens[var]
            if var_lev == 1:
                out_scale.append(self.output_scale[var].values)
            else:
                for i in range(var_lev):
                    out_scale.append(self.output_scale[var].values[i])
        out_scale = np.array(out_scale)
        if write:
            np.savetxt(save_path + '/out_scale.txt', out_scale.reshape(1, -1), fmt=fmt, delimiter=',')
        return input_sub, input_div, out_scale

    @staticmethod
    def ls(dir_path = ''):
        '''
        You can treat this as a Python wrapper for the bash command "ls".
        '''
        return os.popen(' '.join(['ls', dir_path])).read().splitlines()
    
    @staticmethod
    def set_plot_params():
        '''
        This function sets the plot parameters for matplotlib.
        '''
        plt.close('all')
        plt.rcParams.update(plt.rcParamsDefault)
        plt.rc('font', family='sans')
        plt.rcParams.update({'font.size': 32,
                            'lines.linewidth': 2,
                            'axes.labelsize': 32,
                            'axes.titlesize': 32,
                            'xtick.labelsize': 32,
                            'ytick.labelsize': 32,
                            'legend.fontsize': 32,
                            'axes.linewidth': 2,
                            "pgf.texsystem": "pdflatex"
                            })
        # %config InlineBackend.figure_format = 'retina'
        # use the above line when working in a jupyter notebook

    @staticmethod
    def load_npy_file(load_path = ''):
        '''
        This function loads the prediction .npy file.
        '''
        with open(load_path, 'rb') as f:
            pred = np.load(f)
        return pred
    
    @staticmethod
    def load_h5_file(load_path = ''):
        '''
        This function loads the prediction .h5 file.
        '''
        hf = h5py.File(load_path, 'r')
        pred = np.array(hf.get('pred'))
        return pred

    def zonal_bin_weight_2d(self, npy_array):
        '''
        This function returns the zonal mean of a 2D numpy array, 
        assuming that the 0th dimension corresponds to time, 
        and the 1st dimension corresponds to the latitude bin
        '''
        npy_array = npy_array * self.grid_info_area[None, :] * self.lat_bin_area_divs[None, :]
        return np.stack([np.sum(npy_array[:, self.lat_bin_dict[lat_bin]], axis = 1) for lat_bin in self.lat_bin_dict.keys()], axis = 1)

    def zonal_bin_weight_3d(self, npy_array):
        '''
        This function returns the zonal mean of a 3D numpy array, 
        assuming that the 0th dimension corresponds to time, 
        the 1st dimension corresponds to the latitude bin,
        and the 2nd dimension corresponds to vertical level
        '''
        npy_array = npy_array * self.grid_info_area[None, :, None] * self.lat_bin_area_divs[None, :, None]
        return np.stack([np.sum(npy_array[:, self.lat_bin_dict[lat_bin], :], axis = 1) for lat_bin in self.lat_bin_dict.keys()], axis = 1)

    def set_pressure_grid(self, data_split):
        '''
        This function sets the pressure weighting for metrics.
        '''
        assert data_split in ['train', 'val', 'scoring', 'test'], 'Provided data_split is not valid. Available options are train, val, scoring, and test.'

        if data_split == 'train':
            assert self.input_train is not None
            state_ps = self.input_train[:,self.ps_index]
            if self.normalize:
                state_ps = state_ps*(self.input_max['state_ps'].values - self.input_min['state_ps'].values) + self.input_mean['state_ps'].values
            state_ps = np.reshape(state_ps, (-1, self.num_latlon))
            pressure_grid_p1 = np.array(self.grid_info['P0']*self.grid_info['hyai'])[:,np.newaxis,np.newaxis]
            pressure_grid_p2 = self.grid_info['hybi'].values[:, np.newaxis, np.newaxis] * state_ps[np.newaxis, :, :]
            self.pressure_grid_train = pressure_grid_p1 + pressure_grid_p2
            self.dp_train = self.pressure_grid_train[1:61,:,:] - self.pressure_grid_train[0:60,:,:]
            self.dp_train = self.dp_train.transpose((1,2,0))
        elif data_split == 'val':
            assert self.input_val is not None
            state_ps = self.input_val[:,self.ps_index]
            if self.normalize:
                state_ps = state_ps*(self.input_max['state_ps'].values - self.input_min['state_ps'].values) + self.input_mean['state_ps'].values
            state_ps = np.reshape(state_ps, (-1, self.num_latlon))
            pressure_grid_p1 = np.array(self.grid_info['P0']*self.grid_info['hyai'])[:,np.newaxis,np.newaxis]
            pressure_grid_p2 = self.grid_info['hybi'].values[:, np.newaxis, np.newaxis] * state_ps[np.newaxis, :, :]
            self.pressure_grid_val = pressure_grid_p1 + pressure_grid_p2
            self.dp_val = self.pressure_grid_val[1:61,:,:] - self.pressure_grid_val[0:60,:,:]
            self.dp_val = self.dp_val.transpose((1,2,0))
        elif data_split == 'scoring':
            assert self.input_scoring is not None
            state_ps = self.input_scoring[:,self.ps_index]
            if self.normalize:
                state_ps = state_ps*(self.input_max['state_ps'].values - self.input_min['state_ps'].values) + self.input_mean['state_ps'].values
            state_ps = np.reshape(state_ps, (-1, self.num_latlon))
            pressure_grid_p1 = np.array(self.grid_info['P0']*self.grid_info['hyai'])[:,np.newaxis,np.newaxis]
            pressure_grid_p2 = self.grid_info['hybi'].values[:, np.newaxis, np.newaxis] * state_ps[np.newaxis, :, :]
            self.pressure_grid_scoring = pressure_grid_p1 + pressure_grid_p2
            self.dp_scoring = self.pressure_grid_scoring[1:61,:,:] - self.pressure_grid_scoring[0:60,:,:]
            self.dp_scoring = self.dp_scoring.transpose((1,2,0))
        elif data_split == 'test':
            assert self.input_test is not None
            state_ps = self.input_test[:,self.ps_index]
            if self.normalize:
                state_ps = state_ps*(self.input_max['state_ps'].values - self.input_min['state_ps'].values) + self.input_mean['state_ps'].values
            state_ps = np.reshape(state_ps, (-1, self.num_latlon))
            pressure_grid_p1 = np.array(self.grid_info['P0']*self.grid_info['hyai'])[:,np.newaxis,np.newaxis]
            pressure_grid_p2 = self.grid_info['hybi'].values[:, np.newaxis, np.newaxis] * state_ps[np.newaxis, :, :]
            self.pressure_grid_test = pressure_grid_p1 + pressure_grid_p2
            self.dp_test = self.pressure_grid_test[1:61,:,:] - self.pressure_grid_test[0:60,:,:]
            self.dp_test = self.dp_test.transpose((1,2,0))

    def output_weighting(self, output, data_split, just_weights = False):
        '''
        This function does four transformations, and assumes we are using V1 variables:
        [0] Undos the output scaling
        [1] Weight vertical levels by dp/g
        [2] Weight horizontal area of each grid cell by a[x]/mean(a[x])
        [3] Unit conversion to a common energy unit
        '''
        assert data_split in ['train', 'val', 'scoring', 'test'], 'Provided data_split is not valid. Available options are train, val, scoring, and test.'
        num_samples = output.shape[0]
        if just_weights:
            weightings = np.ones(output.shape)

        if not self.full_vars and not self.microphysics_constraint:
            ptend_t = output[:,:60].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
            ptend_q0001 = output[:,60:120].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
            netsw = output[:,120].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            flwds = output[:,121].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            precsc = output[:,122].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            precc = output[:,123].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            sols = output[:,124].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            soll = output[:,125].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            solsd = output[:,126].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            solld = output[:,127].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            if just_weights:
                ptend_t_weight = weightings[:,:60].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
                ptend_q0001_weight = weightings[:,60:120].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
                netsw_weight = weightings[:,360].reshape((int(num_samples/self.num_latlon), self.num_latlon))
                flwds_weight = weightings[:,361].reshape((int(num_samples/self.num_latlon), self.num_latlon))
                precsc_weight = weightings[:,362].reshape((int(num_samples/self.num_latlon), self.num_latlon))
                precc_weight = weightings[:,363].reshape((int(num_samples/self.num_latlon), self.num_latlon))
                sols_weight = weightings[:,364].reshape((int(num_samples/self.num_latlon), self.num_latlon))
                soll_weight = weightings[:,365].reshape((int(num_samples/self.num_latlon), self.num_latlon))
                solsd_weight = weightings[:,366].reshape((int(num_samples/self.num_latlon), self.num_latlon))
                solld_weight = weightings[:,367].reshape((int(num_samples/self.num_latlon), self.num_latlon))
        elif self.microphysics_constraint:
            ptend_t = output[:,:60].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
            ptend_q0001 = output[:,60:120].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
            ptend_qn = output[:,120:180].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
            ptend_u = output[:,180:240].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
            ptend_v = output[:,240:300].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
            netsw = output[:,300].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            flwds = output[:,301].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            precsc = output[:,302].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            precc = output[:,303].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            sols = output[:,304].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            soll = output[:,305].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            solsd = output[:,306].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            solld = output[:,307].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            state_wind = ((ptend_u**2) + (ptend_v**2))**.5
            self.target_energy_conv['ptend_wind'] = state_wind
            if just_weights:
                ptend_t_weight = weightings[:,:60].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
                ptend_q0001_weight = weightings[:,60:120].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
                ptend_qn_weight = weightings[:,120:180].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
                ptend_u_weight = weightings[:,180:240].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
                ptend_v_weight = weightings[:,240:300].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
                netsw_weight = weightings[:,300].reshape((int(num_samples/self.num_latlon), self.num_latlon))
                flwds_weight = weightings[:,301].reshape((int(num_samples/self.num_latlon), self.num_latlon))
                precsc_weight = weightings[:,302].reshape((int(num_samples/self.num_latlon), self.num_latlon))
                precc_weight = weightings[:,303].reshape((int(num_samples/self.num_latlon), self.num_latlon))
                sols_weight = weightings[:,304].reshape((int(num_samples/self.num_latlon), self.num_latlon))
                soll_weight = weightings[:,305].reshape((int(num_samples/self.num_latlon), self.num_latlon))
                solsd_weight = weightings[:,306].reshape((int(num_samples/self.num_latlon), self.num_latlon))
                solld_weight = weightings[:,307].reshape((int(num_samples/self.num_latlon), self.num_latlon))
        else:
            ptend_t = output[:,:60].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
            ptend_q0001 = output[:,60:120].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
            ptend_q0002 = output[:,120:180].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
            ptend_q0003 = output[:,180:240].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
            ptend_u = output[:,240:300].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
            ptend_v = output[:,300:360].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
            netsw = output[:,360].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            flwds = output[:,361].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            precsc = output[:,362].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            precc = output[:,363].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            sols = output[:,364].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            soll = output[:,365].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            solsd = output[:,366].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            solld = output[:,367].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            state_wind = ((ptend_u**2) + (ptend_v**2))**.5
            self.target_energy_conv['ptend_wind'] = state_wind
            if just_weights:
                ptend_t_weight = weightings[:,:60].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
                ptend_q0001_weight = weightings[:,60:120].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
                ptend_q0002_weight = weightings[:,120:180].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
                ptend_q0003_weight = weightings[:,180:240].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
                ptend_u_weight = weightings[:,240:300].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
                ptend_v_weight = weightings[:,300:360].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
                netsw_weight = weightings[:,360].reshape((int(num_samples/self.num_latlon), self.num_latlon))
                flwds_weight = weightings[:,361].reshape((int(num_samples/self.num_latlon), self.num_latlon))
                precsc_weight = weightings[:,362].reshape((int(num_samples/self.num_latlon), self.num_latlon))
                precc_weight = weightings[:,363].reshape((int(num_samples/self.num_latlon), self.num_latlon))
                sols_weight = weightings[:,364].reshape((int(num_samples/self.num_latlon), self.num_latlon))
                soll_weight = weightings[:,365].reshape((int(num_samples/self.num_latlon), self.num_latlon))
                solsd_weight = weightings[:,366].reshape((int(num_samples/self.num_latlon), self.num_latlon))
                solld_weight = weightings[:,367].reshape((int(num_samples/self.num_latlon), self.num_latlon))
            
        # ptend_t = ptend_t.transpose((2,0,1))
        # ptend_q0001 = ptend_q0001.transpose((2,0,1))
        # scalar_outputs = scalar_outputs.transpose((2,0,1))

        # [0] Undo output scaling
        if self.normalize:
            ptend_t = ptend_t/self.output_scale['ptend_t'].values[np.newaxis, np.newaxis, :]
            ptend_q0001 = ptend_q0001/self.output_scale['ptend_q0001'].values[np.newaxis, np.newaxis, :]
            netsw = netsw/self.output_scale['cam_out_NETSW'].values
            flwds = flwds/self.output_scale['cam_out_FLWDS'].values
            precsc = precsc/self.output_scale['cam_out_PRECSC'].values
            precc = precc/self.output_scale['cam_out_PRECC'].values
            sols = sols/self.output_scale['cam_out_SOLS'].values
            soll = soll/self.output_scale['cam_out_SOLL'].values
            solsd = solsd/self.output_scale['cam_out_SOLSD'].values
            solld = solld/self.output_scale['cam_out_SOLLD'].values
            if just_weights:
                ptend_t_weight = ptend_t_weight/self.output_scale['ptend_t'].values[np.newaxis, np.newaxis, :]
                ptend_q0001_weight = ptend_q0001_weight/self.output_scale['ptend_q0001'].values[np.newaxis, np.newaxis, :]
                netsw_weight = netsw_weight/self.output_scale['cam_out_NETSW'].values
                flwds_weight = flwds_weight/self.output_scale['cam_out_FLWDS'].values
                precsc_weight = precsc_weight/self.output_scale['cam_out_PRECSC'].values
                precc_weight = precc_weight/self.output_scale['cam_out_PRECC'].values
                sols_weight = sols_weight/self.output_scale['cam_out_SOLS'].values
                soll_weight = soll_weight/self.output_scale['cam_out_SOLL'].values
                solsd_weight = solsd_weight/self.output_scale['cam_out_SOLSD'].values
                solld_weight = solld_weight/self.output_scale['cam_out_SOLLD'].values
            if self.full_vars:
                ptend_q0002 = ptend_q0002/self.output_scale['ptend_q0002'].values[np.newaxis, np.newaxis, :]
                ptend_q0003 = ptend_q0003/self.output_scale['ptend_q0003'].values[np.newaxis, np.newaxis, :]
                ptend_u = ptend_u/self.output_scale['ptend_u'].values[np.newaxis, np.newaxis, :]
                ptend_v = ptend_v/self.output_scale['ptend_v'].values[np.newaxis, np.newaxis, :]
                if just_weights:
                    ptend_q0002_weight = ptend_q0002_weight/self.output_scale['ptend_q0002'].values[np.newaxis, np.newaxis, :]
                    ptend_q0003_weight = ptend_q0003_weight/self.output_scale['ptend_q0003'].values[np.newaxis, np.newaxis, :]
                    ptend_u_weight = ptend_u_weight/self.output_scale['ptend_u'].values[np.newaxis, np.newaxis, :]
                    ptend_v_weight = ptend_v_weight/self.output_scale['ptend_v'].values[np.newaxis, np.newaxis, :]
            if self.microphysics_constraint:
                ptend_qn = ptend_qn/self.output_scale['ptend_qn'].values[np.newaxis, np.newaxis, :]
                ptend_u = ptend_u/self.output_scale['ptend_u'].values[np.newaxis, np.newaxis, :]
                ptend_v = ptend_v/self.output_scale['ptend_v'].values[np.newaxis, np.newaxis, :]
                if just_weights:
                    ptend_qn_weight = ptend_qn_weight/self.output_scale['ptend_qn'].values[np.newaxis, np.newaxis, :]
                    ptend_u_weight = ptend_u_weight/self.output_scale['ptend_u'].values[np.newaxis, np.newaxis, :]
                    ptend_v_weight = ptend_v_weight/self.output_scale['ptend_v'].values[np.newaxis, np.newaxis, :]
        # [1] Weight vertical levels by dp/g
        # only for vertically-resolved variables, e.g. ptend_{t,q0001}
        # dp/g = -\rho * dz

        dp = None
        if data_split == 'train':
            dp = self.dp_train
        elif data_split == 'val':
            dp = self.dp_val
        elif data_split == 'scoring':
            dp = self.dp_scoring
        elif data_split == 'test':
            dp = self.dp_test
        assert dp is not None
        ptend_t = ptend_t * dp/self.grav
        ptend_q0001 = ptend_q0001 * dp/self.grav
        if just_weights:
            ptend_t_weight = ptend_t_weight * dp/self.grav
            ptend_q0001_weight = ptend_q0001_weight * dp/self.grav
        if self.full_vars:
            ptend_q0002 = ptend_q0002 * dp/self.grav
            ptend_q0003 = ptend_q0003 * dp/self.grav
            ptend_u = ptend_u * dp/self.grav
            ptend_v = ptend_v * dp/self.grav
            if just_weights:
                ptend_q0002_weight = ptend_q0002_weight * dp/self.grav
                ptend_q0003_weight = ptend_q0003_weight * dp/self.grav
                ptend_u_weight = ptend_u_weight * dp/self.grav  
                ptend_v_weight = ptend_v_weight * dp/self.grav
        if self.microphysics_constraint:
            ptend_qn = ptend_qn * dp/self.grav
            ptend_u = ptend_u * dp/self.grav
            ptend_v = ptend_v * dp/self.grav
            if just_weights:
                ptend_qn_weight = ptend_qn_weight * dp/self.grav
                ptend_u_weight = ptend_u_weight * dp/self.grav
                ptend_v_weight = ptend_v_weight * dp/self.grav

        # [2] weight by area

        ptend_t = ptend_t * self.area_wgt[np.newaxis, :, np.newaxis]
        ptend_q0001 = ptend_q0001 * self.area_wgt[np.newaxis, :, np.newaxis]
        netsw = netsw * self.area_wgt[np.newaxis, :]
        flwds = flwds * self.area_wgt[np.newaxis, :]
        precsc = precsc * self.area_wgt[np.newaxis, :]
        precc = precc * self.area_wgt[np.newaxis, :]
        sols = sols * self.area_wgt[np.newaxis, :]
        soll = soll * self.area_wgt[np.newaxis, :]
        solsd = solsd * self.area_wgt[np.newaxis, :]
        solld = solld * self.area_wgt[np.newaxis, :]
        if just_weights:
            ptend_t_weight = ptend_t_weight * self.area_wgt[np.newaxis, :, np.newaxis]
            ptend_q0001_weight = ptend_q0001_weight * self.area_wgt[np.newaxis, :, np.newaxis]
            netsw_weight = netsw_weight * self.area_wgt[np.newaxis, :]
            flwds_weight = flwds_weight * self.area_wgt[np.newaxis, :]
            precsc_weight = precsc_weight * self.area_wgt[np.newaxis, :]
            precc_weight = precc_weight * self.area_wgt[np.newaxis, :]
            sols_weight = sols_weight * self.area_wgt[np.newaxis, :]
            soll_weight = soll_weight * self.area_wgt[np.newaxis, :]
            solsd_weight = solsd_weight * self.area_wgt[np.newaxis, :]
            solld_weight = solld_weight * self.area_wgt[np.newaxis, :]
        if self.full_vars:
            ptend_q0002 = ptend_q0002 * self.area_wgt[np.newaxis, :, np.newaxis]
            ptend_q0003 = ptend_q0003 * self.area_wgt[np.newaxis, :, np.newaxis]
            ptend_u = ptend_u * self.area_wgt[np.newaxis, :, np.newaxis]
            ptend_v = ptend_v * self.area_wgt[np.newaxis, :, np.newaxis]
            if just_weights:
                ptend_q0002_weight = ptend_q0002_weight * self.area_wgt[np.newaxis, :, np.newaxis]
                ptend_q0003_weight = ptend_q0003_weight * self.area_wgt[np.newaxis, :, np.newaxis]
                ptend_u_weight = ptend_u_weight * self.area_wgt[np.newaxis, :, np.newaxis]
                ptend_v_weight = ptend_v_weight * self.area_wgt[np.newaxis, :, np.newaxis]
        if self.microphysics_constraint:
            ptend_qn = ptend_qn * self.area_wgt[np.newaxis, :, np.newaxis]
            ptend_u = ptend_u * self.area_wgt[np.newaxis, :, np.newaxis]
            ptend_v = ptend_v * self.area_wgt[np.newaxis, :, np.newaxis]
            if just_weights:
                ptend_qn_weight = ptend_qn_weight * self.area_wgt[np.newaxis, :, np.newaxis]
                ptend_u_weight = ptend_u_weight * self.area_wgt[np.newaxis, :, np.newaxis]
                ptend_v_weight = ptend_v_weight * self.area_wgt[np.newaxis, :, np.newaxis]

        # [3] unit conversion

        ptend_t = ptend_t * self.target_energy_conv['ptend_t']
        ptend_q0001 = ptend_q0001 * self.target_energy_conv['ptend_q0001']
        netsw = netsw * self.target_energy_conv['cam_out_NETSW']
        flwds = flwds * self.target_energy_conv['cam_out_FLWDS']
        precsc = precsc * self.target_energy_conv['cam_out_PRECSC']
        precc = precc * self.target_energy_conv['cam_out_PRECC']
        sols = sols * self.target_energy_conv['cam_out_SOLS']
        soll = soll * self.target_energy_conv['cam_out_SOLL']
        solsd = solsd * self.target_energy_conv['cam_out_SOLSD']
        solld = solld * self.target_energy_conv['cam_out_SOLLD']
        if just_weights:
            ptend_t_weight = ptend_t_weight * self.target_energy_conv['ptend_t']
            ptend_q0001_weight = ptend_q0001_weight * self.target_energy_conv['ptend_q0001']
            netsw_weight = netsw_weight * self.target_energy_conv['cam_out_NETSW']
            flwds_weight = flwds_weight * self.target_energy_conv['cam_out_FLWDS']
            precsc_weight = precsc_weight * self.target_energy_conv['cam_out_PRECSC']
            precc_weight = precc_weight * self.target_energy_conv['cam_out_PRECC']
            sols_weight = sols_weight * self.target_energy_conv['cam_out_SOLS']
            soll_weight = soll_weight * self.target_energy_conv['cam_out_SOLL']
            solsd_weight = solsd_weight * self.target_energy_conv['cam_out_SOLSD']
            solld_weight = solld_weight * self.target_energy_conv['cam_out_SOLLD']
        if self.full_vars:
            ptend_q0002 = ptend_q0002 * self.target_energy_conv['ptend_q0002']
            ptend_q0003 = ptend_q0003 * self.target_energy_conv['ptend_q0003']
            ptend_u = ptend_u * self.target_energy_conv['ptend_wind']
            ptend_v = ptend_v * self.target_energy_conv['ptend_wind']
            if just_weights:
                ptend_q0002_weight = ptend_q0002_weight * self.target_energy_conv['ptend_q0002']
                ptend_q0003_weight = ptend_q0003_weight * self.target_energy_conv['ptend_q0003']
                ptend_u_weight = ptend_u_weight * self.target_energy_conv['ptend_wind']
                ptend_v_weight = ptend_v_weight * self.target_energy_conv['ptend_wind']
        if self.microphysics_constraint:
            ptend_qn = ptend_qn * self.target_energy_conv['ptend_qn']
            ptend_u = ptend_u * self.target_energy_conv['ptend_wind']
            ptend_v = ptend_v * self.target_energy_conv['ptend_wind']
            if just_weights:
                ptend_qn_weight = ptend_qn_weight * self.target_energy_conv['ptend_qn']
                ptend_u_weight = ptend_u_weight * self.target_energy_conv['ptend_wind']
                ptend_v_weight = ptend_v_weight * self.target_energy_conv['ptend_wind']


        if just_weights:
            if self.full_vars:
                weightings = np.concatenate([ptend_t_weight.reshape((num_samples, 60)), \
                                             ptend_q0001_weight.reshape((num_samples, 60)), \
                                             ptend_q0002_weight.reshape((num_samples, 60)), \
                                             ptend_q0003_weight.reshape((num_samples, 60)), \
                                             ptend_u_weight.reshape((num_samples, 60)), \
                                             ptend_v_weight.reshape((num_samples, 60)), \
                                             netsw_weight.reshape((num_samples))[:, np.newaxis], \
                                             flwds_weight.reshape((num_samples))[:, np.newaxis], \
                                             precsc_weight.reshape((num_samples))[:, np.newaxis], \
                                             precc_weight.reshape((num_samples))[:, np.newaxis], \
                                             sols_weight.reshape((num_samples))[:, np.newaxis], \
                                             soll_weight.reshape((num_samples))[:, np.newaxis], \
                                             solsd_weight.reshape((num_samples))[:, np.newaxis], \
                                             solld_weight.reshape((num_samples))[:, np.newaxis]], axis = 1)
            elif self.microphysics_constraint:
                weightings = np.concatenate([ptend_t_weight.reshape((num_samples, 60)), \
                                             ptend_q0001_weight.reshape((num_samples, 60)), \
                                             ptend_qn_weight.reshape((num_samples, 60)), \
                                             ptend_u_weight.reshape((num_samples, 60)), \
                                             ptend_v_weight.reshape((num_samples, 60)), \
                                             netsw_weight.reshape((num_samples))[:, np.newaxis], \
                                             flwds_weight.reshape((num_samples))[:, np.newaxis], \
                                             precsc_weight.reshape((num_samples))[:, np.newaxis], \
                                             precc_weight.reshape((num_samples))[:, np.newaxis], \
                                             sols_weight.reshape((num_samples))[:, np.newaxis], \
                                             soll_weight.reshape((num_samples))[:, np.newaxis], \
                                             solsd_weight.reshape((num_samples))[:, np.newaxis], \
                                             solld_weight.reshape((num_samples))[:, np.newaxis]], axis = 1)
            else:
                weightings = np.concatenate([ptend_t_weight.reshape((num_samples, 60)), \
                                             ptend_q0001_weight.reshape((num_samples, 60)), \
                                             netsw_weight.reshape((num_samples))[:, np.newaxis], \
                                             flwds_weight.reshape((num_samples))[:, np.newaxis], \
                                             precsc_weight.reshape((num_samples))[:, np.newaxis], \
                                             precc_weight.reshape((num_samples))[:, np.newaxis], \
                                             sols_weight.reshape((num_samples))[:, np.newaxis], \
                                             soll_weight.reshape((num_samples))[:, np.newaxis], \
                                             solsd_weight.reshape((num_samples))[:, np.newaxis], \
                                             solld_weight.reshape((num_samples))[:, np.newaxis]], axis = 1)
            return weightings
        else:
            var_dict = {'ptend_t':ptend_t,
                        'ptend_q0001':ptend_q0001,
                        'cam_out_NETSW':netsw,
                        'cam_out_FLWDS':flwds,
                        'cam_out_PRECSC':precsc,
                        'cam_out_PRECC':precc,
                        'cam_out_SOLS':sols,
                        'cam_out_SOLL':soll,
                        'cam_out_SOLSD':solsd,
                        'cam_out_SOLLD':solld}
            if self.full_vars:
                var_dict['ptend_q0002'] = ptend_q0002
                var_dict['ptend_q0003'] = ptend_q0003
                var_dict['ptend_u'] = ptend_u
                var_dict['ptend_v'] = ptend_v
            if self.microphysics_constraint:
                var_dict['ptend_qn'] = ptend_qn
                var_dict['ptend_u'] = ptend_u
                var_dict['ptend_v'] = ptend_v

            return var_dict

    def reweight_target(self, data_split):
        '''
        data_split should be train, val, scoring, or test
        weights target variables assuming V1 outputs using the output_weighting function
        '''
        assert data_split in ['train', 'val', 'scoring', 'test'], 'Provided data_split is not valid. Available options are train, val, scoring, and test.'
        if data_split == 'train':
            assert self.target_train is not None
            self.target_weighted_train = self.output_weighting(self.target_train, data_split)
        elif data_split == 'val':
            assert self.target_val is not None
            self.target_weighted_val = self.output_weighting(self.target_val, data_split)
        elif data_split == 'scoring':
            assert self.target_scoring is not None
            self.target_weighted_scoring = self.output_weighting(self.target_scoring, data_split)
        elif data_split == 'test':
            assert self.target_test is not None
            self.target_weighted_test = self.output_weighting(self.target_test, data_split)

    def reweight_preds(self, data_split):
        '''
        weights predictions assuming V1 outputs using the output_weighting function
        '''
        assert data_split in ['train', 'val', 'scoring', 'test'], 'Provided data_split is not valid. Available options are train, val, scoring, and test.'
        assert self.model_names is not None

        if data_split == 'train':
            assert self.preds_train is not None
            for model_name in self.model_names:
                self.preds_weighted_train[model_name] = self.output_weighting(self.preds_train[model_name], data_split)
        elif data_split == 'val':
            assert self.preds_val is not None
            for model_name in self.model_names:
                self.preds_weighted_val[model_name] = self.output_weighting(self.preds_val[model_name], data_split)
        elif data_split == 'scoring':
            assert self.preds_scoring is not None
            for model_name in self.model_names:
                self.preds_weighted_scoring[model_name] = self.output_weighting(self.preds_scoring[model_name], data_split)
        elif data_split == 'test':
            assert self.preds_test is not None
            for model_name in self.model_names:
                self.preds_weighted_test[model_name] = self.output_weighting(self.preds_test[model_name], data_split)

    def reweight_samplepreds(self, data_split):
        '''
        weights predictions assuming V1 outputs using the output_weighting function
        need to edit to get it to work across samples
        '''
        assert data_split in ['train', 'val', 'scoring', 'test'], 'Provided data_split is not valid. Available options are train, val, scoring, and test.'
        assert self.model_names is not None

        if data_split == 'train':
            assert self.samplepreds_train is not None
            for model_name in self.model_names:
                self.samplepreds_weighted_train[model_name] = self.output_weighting_CRPS(self.samplepreds_train[model_name], data_split)
        elif data_split == 'val':
            assert self.samplepreds_val is not None
            for model_name in self.model_names:
                self.samplepreds_weighted_val[model_name] = self.output_weighting_CRPS(self.samplepreds_val[model_name], data_split)
        elif data_split == 'scoring':
            assert self.samplepreds_scoring is not None
            for model_name in self.model_names:
                self.samplepreds_weighted_scoring[model_name] = self.output_weighting_CRPS(self.samplepreds_scoring[model_name], data_split)
        elif data_split == 'test':
            assert self.samplepreds_test is not None
            for model_name in self.model_names:
                self.samplepreds_weighted_test[model_name] = self.output_weighting_CRPS(self.samplepreds_test[model_name], data_split)

    def calc_MAE(self, pred, target, avg_grid = True):
        '''
        calculate 'globally averaged' mean absolute error 
        for vertically-resolved variables, shape should be time x grid x level
        for scalars, shape should be time x grid

        returns vector of length level or 1
        '''
        assert pred.shape[1] == self.num_latlon
        assert pred.shape == target.shape
        mae = np.abs(pred - target).mean(axis = 0)
        if avg_grid:
            return mae.mean(axis = 0) # we decided to average globally at end
        else:
            return mae
    
    def calc_RMSE(self, pred, target, avg_grid = True):
        '''
        calculate 'globally averaged' root mean squared error 
        for vertically-resolved variables, shape should be time x grid x level
        for scalars, shape should be time x grid

        returns vector of length level or 1
        '''
        assert pred.shape[1] == self.num_latlon
        assert pred.shape == target.shape
        sq_diff = (pred - target)**2
        rmse = np.sqrt(sq_diff.mean(axis = 0)) # mean over time
        if avg_grid:
            return rmse.mean(axis = 0) # we decided to separately average globally at end
        else:
            return rmse

    def calc_R2(self, pred, target, avg_grid = True):
        '''
        calculate 'globally averaged' R-squared
        for vertically-resolved variables, input shape should be time x grid x level
        for scalars, input shape should be time x grid

        returns vector of length level or 1
        '''
        assert pred.shape[1] == self.num_latlon
        assert pred.shape == target.shape
        sq_diff = (pred - target)**2
        tss_time = (target - target.mean(axis = 0)[np.newaxis, ...])**2 # mean over time
        r_squared = 1 - sq_diff.sum(axis = 0)/tss_time.sum(axis = 0) # sum over time
        if avg_grid:
            return r_squared.mean(axis = 0) # we decided to separately average globally at end
        else:
            return r_squared
    
    def calc_bias(self, pred, target, avg_grid = True):
        '''
        calculate bias
        for vertically-resolved variables, input shape should be time x grid x level
        for scalars, input shape should be time x grid

        returns vector of length level or 1
        '''
        assert pred.shape[1] == self.num_latlon
        assert pred.shape == target.shape
        bias = pred.mean(axis = 0) - target.mean(axis = 0)
        if avg_grid:
            return bias.mean(axis = 0) # we decided to separately average globally at end
        else:
            return bias
        
    def calc_CRPS(self, samplepreds, target, avg_grid = True):
        '''
        calculate 'globally averaged' continuous ranked probability score
        for vertically-resolved variables, input shape should be time x grid x level x num_crps_samples
        for scalars, input shape should be time x grid x num_crps_samples

        returns vector of length level or 1
        '''
        assert samplepreds.shape[1] == self.num_latlon
        num_crps = samplepreds.shape[-1]
        mae = np.mean(np.abs(samplepreds - target[..., np.newaxis]), axis = (0, -1)) # mean over time and crps samples
        diff = samplepreds[..., 1:] - samplepreds[..., :-1]
        count = np.arange(1, num_crps) * np.arange(num_crps - 1, 0, -1)
        spread = (diff * count[np.newaxis, np.newaxis, np.newaxis, :]).mean(axis = (0, -1)) # mean over time and crps samples
        crps = mae - spread/(num_crps*(num_crps-1))
        # already divided by two in spread by exploiting symmetry
        if avg_grid:
            return crps.mean(axis = 0) # we decided to separately average globally at end
        else:
            return crps

    def create_metrics_df(self, data_split):
        '''
        creates a dataframe of metrics for each model
        '''
        assert data_split in ['train', 'val', 'scoring', 'test'], \
            'Provided data_split is not valid. Available options are train, val, scoring, and test.'
        assert len(self.model_names) != 0
        assert len(self.metrics_names) != 0
        assert len(self.target_vars) != 0
        assert self.target_feature_len is not None

        if data_split == 'train':
            assert len(self.preds_weighted_train) != 0
            assert len(self.target_weighted_train) != 0
            for model_name in self.model_names:
                df_var = pd.DataFrame(columns = self.metrics_names, index = self.target_vars)
                df_var.index.name = 'variable'
                df_idx = pd.DataFrame(columns = self.metrics_names, index = range(self.target_feature_len))
                df_idx.index.name = 'output_idx'
                for metric_name in self.metrics_names:
                    current_idx = 0
                    for target_var in self.target_vars:
                        metric = self.metrics_dict[metric_name](self.preds_weighted_train[model_name][target_var], self.target_weighted_train[target_var])
                        df_var.loc[target_var, metric_name] = np.mean(metric)
                        df_idx.loc[current_idx:current_idx + self.var_lens[target_var] - 1, metric_name] = np.atleast_1d(metric)
                        current_idx += self.var_lens[target_var]
                self.metrics_var_train[model_name] = df_var
                self.metrics_idx_train[model_name] = df_idx

        elif data_split == 'val':
            assert len(self.preds_weighted_val) != 0
            assert len(self.target_weighted_val) != 0
            for model_name in self.model_names:
                df_var = pd.DataFrame(columns = self.metrics_names, index = self.target_vars)
                df_var.index.name = 'variable'
                df_idx = pd.DataFrame(columns = self.metrics_names, index = range(self.target_feature_len))
                df_idx.index.name = 'output_idx'
                for metric_name in self.metrics_names:
                    current_idx = 0
                    for target_var in self.target_vars:
                        metric = self.metrics_dict[metric_name](self.preds_weighted_val[model_name][target_var], self.target_weighted_val[target_var])
                        df_var.loc[target_var, metric_name] = np.mean(metric)
                        df_idx.loc[current_idx:current_idx + self.var_lens[target_var] - 1, metric_name] = np.atleast_1d(metric)
                        current_idx += self.var_lens[target_var]
                self.metrics_var_val[model_name] = df_var
                self.metrics_idx_val[model_name] = df_idx

        elif data_split == 'scoring':
            assert len(self.preds_weighted_scoring) != 0
            assert len(self.target_weighted_scoring) != 0
            for model_name in self.model_names:
                df_var = pd.DataFrame(columns = self.metrics_names, index = self.target_vars)
                df_var.index.name = 'variable'
                df_idx = pd.DataFrame(columns = self.metrics_names, index = range(self.target_feature_len))
                df_idx.index.name = 'output_idx'
                for metric_name in self.metrics_names:
                    current_idx = 0
                    for target_var in self.target_vars:
                        metric = self.metrics_dict[metric_name](self.preds_weighted_scoring[model_name][target_var], self.target_weighted_scoring[target_var])
                        df_var.loc[target_var, metric_name] = np.mean(metric)
                        df_idx.loc[current_idx:current_idx + self.var_lens[target_var] - 1, metric_name] = np.atleast_1d(metric)
                        current_idx += self.var_lens[target_var]
                self.metrics_var_scoring[model_name] = df_var
                self.metrics_idx_scoring[model_name] = df_idx

        elif data_split == 'test':
            assert len(self.preds_weighted_test) != 0
            assert len(self.target_weighted_test) != 0
            for model_name in self.model_names:
                df_var = pd.DataFrame(columns = self.metrics_names, index = self.target_vars)
                df_var.index.name = 'variable'
                df_idx = pd.DataFrame(columns = self.metrics_names, index = range(self.target_feature_len))
                df_idx.index.name = 'output_idx'
                for metric_name in self.metrics_names:
                    current_idx = 0
                    for target_var in self.target_vars:
                        metric = self.metrics_dict[metric_name](self.preds_weighted_test[model_name][target_var], self.target_weighted_test[target_var])
                        df_var.loc[target_var, metric_name] = np.mean(metric)
                        df_idx.loc[current_idx:current_idx + self.var_lens[target_var] - 1, metric_name] = np.atleast_1d(metric)
                        current_idx += self.var_lens[target_var]
                self.metrics_var_test[model_name] = df_var
                self.metrics_idx_test[model_name] = df_idx