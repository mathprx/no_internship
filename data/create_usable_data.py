import xarray as xr
from data_utils import data_utils #climsim_utils.
raw_data_folder_path = 'data/raw_data/train/'
usable_data_folder_path = 'data/usable_data'

training_data_paths_regex = '*mli.0002-*-0[1-3]-*.nc' 
validation_data_paths_regex = '*mli.0002-*-15-*.nc'
testing_data_paths_regex = '*mli.0002-*-20-*.nc'

grid_path = 'data/ClimSim_low-res_grid-info.nc'
grid_info = xr.open_dataset(grid_path)

data = data_utils(grid_info,
                 input_mean=None,
                 input_max=None,
                 input_min = None,
                 output_scale = None,
                 qinput_log = False,
                 normalize = False,
                 input_abbrev = 'mli',
                 output_abbrev = 'mlo',
                 save_zarr=False,
                 save_h5=False,
                 save_npy=True,
                 cpuonly=False)

data.set_to_v2_rh_mc_vars()

data.data_path = raw_data_folder_path

data.set_regexps(data_split='train', regexps=[training_data_paths_regex])
data.set_stride_sample(data_split='train', stride_sample=1)
data.set_filelist(data_split='train')
data.set_regexps(data_split='val', regexps=[validation_data_paths_regex])
data.set_stride_sample(data_split='val', stride_sample=1)
data.set_filelist(data_split='val')
data.set_regexps(data_split='test', regexps=[testing_data_paths_regex])
data.set_stride_sample(data_split='test', stride_sample=1)
data.set_filelist(data_split='test')

print('Ready to create usable data from raw data...')

data.save_as_npy(data_split='train', save_path=usable_data_folder_path)
print('Usable data created for training set.')
data.save_as_npy(data_split='val', save_path=usable_data_folder_path)
print('Usable data created for validation set.')
data.save_as_npy(data_split='test', save_path=usable_data_folder_path)
print('Usable data created for testing set.')

