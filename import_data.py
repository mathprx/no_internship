from data_utils import *

grid_path = '/work/FAC/FGSE/IDYST/tbeucler/ai4pex/Physics_no_offline/ClimSim_offline/modules/climsim3/grid_info/ClimSim_low-res_grid-info.nc'
norm_path = '/work/FAC/FGSE/IDYST/tbeucler/ai4pex/Physics_no_offline/ClimSim_offline/modules/climsim3/preprocessing/normalizations/'

grid_info = xr.open_dataset(grid_path)
#no naming issue here. Here these normalization-related files are just placeholders since we set normalize=False in the data_utils.
input_mean = xr.open_dataset(norm_path + 'inputs/input_mean_v5_pervar.nc')
input_max = xr.open_dataset(norm_path + 'inputs/input_max_v5_pervar.nc')
input_min = xr.open_dataset(norm_path + 'inputs/input_min_v5_pervar.nc')
output_scale = xr.open_dataset(norm_path + 'outputs/output_scale_std_lowerthred_v5.nc')

#print(input_mean.values())

data = data_utils(grid_info = grid_info, 
                  input_mean = input_mean, 
                  input_max = input_max, 
                  input_min = input_min, 
                  output_scale = output_scale,
                  qinput_log = False , # added by me
                  input_abbrev = 'mlexpand',
                  output_abbrev = 'mlo',
                  normalize=False,
                  save_h5=True,
                  save_npy=True
                  )


# set data path
data.data_path = '/work/FAC/FGSE/IDYST/tbeucler/ai4pex/Physics_no_offline/ClimSim_offline/raw_data/train/'
#data.data_path = '/work/FAC/FGSE/IDYST/tbeucler/ai4pex/Physics_no_offline/ClimSim_offline/raw_data/'


# set inputs and outputs to V2 rh subset (rh means using RH to replace specific humidty in input feature)
data.set_to_v2_rh_vars()

# set regular expressions for selecting training data
data.set_regexps(data_split = 'train', 
                regexps = ['E3SM-MMF.mlexpand.000[1234567]-*-*-*.nc', # years 1 through 7
                        'E3SM-MMF.mlexpand.0008-01-*-*.nc']) # first month of year 8

# set temporal subsampling
data.set_stride_sample(data_split = 'train', stride_sample = 1000)
# create list of files to extract data from
data.set_filelist(data_split = 'train', start_idx=0)
#print('PPPPPPPPPPPP',data.get_filelist(data_split='train'))
# save numpy files of training data
data.save_as_npy(data_split = 'train', save_path = '/work/FAC/FGSE/IDYST/tbeucler/ai4pex/Physics_no_offline/ClimSim_offline/data/')



# set regular expressions for selecting validation data
data.set_regexps(data_split = 'val',
                 regexps = ['E3SM-MMF.mlexpand.0008-0[23456789]-*-*.nc', # months 2 through 9 of year 8
                            'E3SM-MMF.mlexpand.0008-1[012]-*-*.nc', # months 10 through 12 of year 8
                            'E3SM-MMF.mlexpand.0009-01-*-*.nc']) # first month of year 9
# set temporal subsampling
# data.set_stride_sample(data_split = 'val', stride_sample = 7)
data.set_stride_sample(data_split = 'val', stride_sample = 700)
# create list of files to extract data from
data.set_filelist(data_split = 'val')
# save numpy files of validation data
data.save_as_npy(data_split = 'val', save_path = '/work/FAC/FGSE/IDYST/tbeucler/ai4pex/Physics_no_offline/ClimSim_offline/data/')



data.data_path = '/work/FAC/FGSE/IDYST/tbeucler/ai4pex/container/climsim-container/storage/ClimSim_low-res-expanded/train/'

data.set_to_v4_vars()

# set regular expressions for selecting validation data
data.set_regexps(data_split = 'test',
                 regexps = ['E3SM-MMF.mlexpand.0009-0[3456789]-*-*.nc', 
                            'E3SM-MMF.mlexpand.0009-1[012]-*-*.nc',
                            'E3SM-MMF.mlexpand.0010-*-*-*.nc',
                            'E3SM-MMF.mlexpand.0011-0[12]-*-*.nc'])
# set temporal subsampling
# data.set_stride_sample(data_split = 'test', stride_sample = 7)
data.set_stride_sample(data_split = 'test', stride_sample = 700)
# create list of files to extract data from
data.set_filelist(data_split = 'test')
# save numpy files of validation data
data.save_as_npy(data_split = 'test', save_path = '/work/FAC/FGSE/IDYST/tbeucler/ai4pex/Physics_no_offline/ClimSim_offline/data/')
