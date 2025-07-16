import numpy as np
import xarray as xr
from data_utils import data_utils  # climsim_utils.

# We first create a data_utils object to get the number of vatibles and their names, but it has no other purpose here.
grid_path = 'data/ClimSim_low-res_grid-info.nc'
grid_info = xr.open_dataset(grid_path)
data_utils = data_utils(grid_info,
                 input_mean=None,
                 input_max=None,
                 input_min=None,
                 output_scale=None,
                 qinput_log=False,
                 normalize=False,
                 input_abbrev=None,
                 output_abbrev=None,
                 save_zarr=False,
                 save_h5=False,
                 save_npy=False,
                 cpuonly=False) 
data_utils.set_to_v2_rh_mc_vars()
input_profile_num = data_utils.input_profile_num
input_scalar_num = data_utils.input_scalar_num
target_profile_num = data_utils.target_profile_num
target_scalar_num = data_utils.target_scalar_num
input_list = data_utils.input_vars
target_list = data_utils.target_vars

print(f'Input profile number: {input_profile_num}')
print(f'Input scalar number: {input_scalar_num}')
print(f'Target profile number: {target_profile_num}')
print(f'Target scalar number: {target_scalar_num}')

input_train_npy_path = 'data/usable_data/train_input.npy'
target_train_npy_path = 'data/usable_data/train_target.npy'

input_array = np.load(input_train_npy_path)
target_array = np.load(target_train_npy_path)

print(f'Input array shape: {input_array.shape}')
print(f'Target array shape: {target_array.shape}')

mean_input = np.zeros(input_profile_num * 60 + input_scalar_num)
max_input = np.zeros(input_profile_num * 60  + input_scalar_num)
min_input = np.zeros(input_profile_num * 60 + input_scalar_num)

mean_input_dict = {}
max_input_dict = {}
min_input_dict = {}
levels = np.arange(60)

for i in range(input_profile_num) :
    variable_name = input_list[i]
    variable_array = input_array[:, i*60:(i+1)*60]
    mean_input[i*60 : (i+1)*60] = np.mean(variable_array)
    mean_input_dict[variable_name] = (["lev"], mean_input[i*60 : (i+1)*60])
    max_input[i*60 : (i+1)*60] = np.max(variable_array)
    mean_input_dict[variable_name] = (["lev"], mean_input[i*60 : (i+1)*60])
    min_input[i*60 : (i+1)*60] = np.min(variable_array)
    mean_input_dict[variable_name] = (["lev"], mean_input[i*60 : (i+1)*60])

for i in range(input_scalar_num) :
    variable_name = input_list[input_profile_num + i]
    variable_array = input_array[:, input_profile_num*60 + i]
    mean_input[input_profile_num + i] = np.mean(variable_array)
    mean_input_dict[variable_name] = ([], mean_input[input_profile_num + i])
    max_input[input_profile_num + i] = np.max(variable_array)
    mean_input_dict[variable_name] = ([], mean_input[input_profile_num + i])
    min_input[input_profile_num + i] = np.min(variable_array)
    mean_input_dict[variable_name] = ([], mean_input[input_profile_num + i])

ds_mean_input = xr.Dataset(mean_input_dict, coords={"lev": levels})
ds_max_input = xr.Dataset(max_input_dict, coords={"lev": levels})
ds_min_input = xr.Dataset(min_input_dict, coords={"lev": levels})

std_target = np.zeros(target_profile_num * 60 + target_scalar_num)

std_target_dict = {}

for i in range(target_profile_num) :
    variable_name = target_list[i]
    variable_array = target_array[:, i*60:(i+1)*60]
    std_target[i*60 : (i+1)*60] = np.std(variable_array)
    std_target_dict[variable_name] = (["lev"], std_target[i*60 : (i+1)*60])

for i in range(target_scalar_num) :
    variable_name = target_list[target_profile_num + i]
    variable_array = target_array[:, target_profile_num*60 + i]
    std_target[target_profile_num + i] = np.std(variable_array)
    std_target_dict[variable_name] = ([], std_target[target_profile_num + i])

ds_std_target = xr.Dataset(std_target_dict, coords={"lev": levels})

# Save the normalization files
save_path = 'data/normalization'

ds_mean_input.to_netcdf(f'{save_path}/mean_input.nc')
ds_max_input.to_netcdf(f'{save_path}/max_input.nc')
ds_min_input.to_netcdf(f'{save_path}/min_input.nc')

ds_std_target.to_netcdf(f'{save_path}/std_target.nc')

np.save(f'{save_path}/mean_input.npy', mean_input)
np.save(f'{save_path}/max_input.npy', max_input)
np.save(f'{save_path}/min_input.npy', min_input)
np.save(f'{save_path}/std_target.npy', std_target)
print('Normalization files created and saved successfully.')