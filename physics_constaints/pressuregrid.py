import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

file_path = "example_job_submit_nnwrapper_v4_constrained.eam.h0.0002-12.nc"
ds = xr.open_dataset(file_path)
# for var in ds.variables :
#     print(var)

def P_surf(i_column):
    return ds['PS'].values[0,i_column]

def dP(i_column) :
    A = ds['hyai'].values
    B = ds['hybi'].values
    P0 = ds['P0'].values
    return np.diff(A*P0 + B*P_surf(i_column))

def P_int(i_column) :
    A = ds['hyai'].values
    B = ds['hybi'].values
    P0 = ds['P0'].values
    return A*P0 + B*P_surf(i_column)

def P_mid(i_column) :
    A = ds['hybm'].values
    B = ds['hybm'].values
    P0 = ds['P0'].values
    print(P0,P_surf(i_column))
    return A*0 + B*P_surf(i_column)

i_column = 0  # Example column index
pressure_mid = P_mid(i_column)
pressure_int = P_int(i_column)

y_int = np.ones(len(pressure_int))
y_mid = np.ones(len(pressure_mid))


plt.scatter(pressure_mid, y_mid, label='P_mid', color='red')
#plt.scatter(pressure_int, y_int, label='P_int', color='blue')
plt.xlabel('Index')
plt.ylabel('Pressure (Pa)')
plt.title('Pressure Profiles')
plt.legend()
plt.grid()
plt.savefig('pressure_profiles.png')