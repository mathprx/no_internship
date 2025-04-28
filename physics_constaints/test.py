import os
import glob
import xarray as xr
import numpy as np
import json
import csv
import warnings

# Counter


# Define the directory to search
directory = '/work/FAC/FGSE/IDYST/tbeucler/ai4pex/Physics_no_offline/no_internship/physics_constaints/run'

# Retrieve all .nc files in the directory and its subdirectories
nc_files = glob.glob(os.path.join(directory, '**', '*.nc'), recursive=True)

var_dic = {}
error_count = 0

serialization_warning_count = 0
def count_serialization_warning(message, category, filename, lineno, file=None, line=None):
    global serialization_warning_count
    serialization_warning_count += 1


for path in nc_files:
    try :
        with warnings.catch_warnings():
            warnings.simplefilter("always")  # Catch all warnings
            warnings.showwarning = count_serialization_warning
            ds = xr.open_dataset(path)
            for var_name, var_data in ds.variables.items():
                long_name = var_data.attrs.get('long_name', 'No long_name attribute')
                var_dic[var_name] = long_name
    except Exception as e:
        error_count += 1  

print('Number of serialization warnings:', serialization_warning_count)
print('Number of errors :', error_count)
print('Number variables :', len(var_dic))
print('Number of files :',len(nc_files))

with open('var_dic.json', 'w') as f:
    json.dump(var_dic, f)

with open("var_dic.csv", "w", newline='') as f:
    writer = csv.writer(f)
    for key, value in var_dic.items():
        writer.writerow([f"{key}: {value}"])