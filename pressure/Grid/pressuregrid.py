import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

file_path = "example_job_submit_nnwrapper_v4_constrained.eam.h0.0002-12.nc"
ds = xr.open_dataset(file_path)
g = 9.81

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
    A = ds['hyam'].values
    B = ds['hybm'].values
    P0 = ds['P0'].values
    print(P0,P_surf(i_column))
    return (A*P0 + B*P_surf(i_column))

i_column = 0  # Example column index
pressure_mid = P_mid(i_column)
pressure_int = P_int(i_column)
print(pressure_int[0], pressure_int[-1])

y_int = np.ones(len(pressure_int))
y_mid = np.ones(len(pressure_mid))

# print('int  :', len(pressure_int))
# print('mid  :', len(pressure_mid))
# print(pressure_mid[:23])

# plt.scatter(y_mid, pressure_mid, label='P_mid', color='red')
# plt.scatter(y_int, pressure_int, label='P_int', color='blue')
# plt.xlabel('Index')
# plt.ylabel('Pressure (Pa)')
# plt.title('Pressure Profiles')
# plt.legend()
# plt.grid()
# plt.show()
# plt.plot(pressure_mid, label='P_mid', color='red')
# plt.plot(pressure_int, label='P_int', color='blue')
# plt.xlabel('Index')
# plt.ylabel('Pressure (Pa)')
# plt.title('Pressure Profiles')
# plt.legend()
# plt.grid()
# plt.show()


# Plot hyai, hybi, hyam, hybm
plt.figure(figsize=(10, 6))

# # Plot hyai and hybi
# plt.plot(ds['hyai'].values, range(len(ds['hyai'].values)), label='hyai', marker='o', linestyle='-', color='blue')
# plt.plot(ds['hybi'].values, range(len(ds['hybi'].values)), label='hybi', marker='o', linestyle='--', color='green')
print(P_surf(0)-ds['P0'].values)
# Plot hyam and hybm
# plt.plot(ds['hyam'].values, range(len(ds['hyam'].values)), label='hyam', marker='x', linestyle='-', color='red')
# plt.plot(ds['hybm'].values, range(len(ds['hybm'].values)), label='hybm', marker='x', linestyle='--', color='orange')
# plt.plot(pressure_mid/P_surf(0), range(len(pressure_mid)), label='P_int', color='blue')

plt.plot(ds['hyai'].values, range(len(ds['hyai'].values)), label='hyai', marker='o', linestyle='-', color='blue')
plt.plot(ds['hybi'].values, range(len(ds['hybi'].values)), label='hybi', marker='o', linestyle='--', color='green')
plt.plot(pressure_int/P_surf(0), range(len(pressure_int)), label='P_int/P_surf', color='red')


plt.ylabel('Index')
plt.xlabel('Value')
plt.title('hyai, hybi, hyam, hybm Profiles')
plt.gca().invert_yaxis()
plt.legend()
plt.grid()
plt.show()
# %%
