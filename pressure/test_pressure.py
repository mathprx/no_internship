import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

grid_path = "/work/FAC/FGSE/IDYST/tbeucler/ai4pex/Physics_no_offline/raw_data/ClimSim_low-res_grid-info_download.nc"
grid_ds = xr.open_dataset(grid_path)

g = 9.81
M = 28.95
kb = 1.380649e-23


def P_surf(i_column,t):
    if t < 10000:
        t = '0' + str(t)
    else:
        t = str(t)
    state_path = "/work/FAC/FGSE/IDYST/tbeucler/ai4pex/Physics_no_offline/raw_data/train/0001-02/E3SM-MMF.mlexpand.0001-02-01-"+t+".nc"
    state_ds = xr.open_dataset(state_path)
    return state_ds['state_ps'].values[i_column]

def dP(i_column,t) :
    A = grid_ds['hyai'].values
    B = grid_ds['hybi'].values
    P0 = grid_ds['P0'].values
    return np.diff(A*P0 + B*P_surf(i_column,t))

def P_int(i_column,t) :
    A = grid_ds['hyai'].values
    B = grid_ds['hybi'].values
    P0 = grid_ds['P0'].values
    return A*P0 + B*P_surf(i_column,t)

def P_mid(i_column,t) :
    A = grid_ds['hyam'].values
    B = grid_ds['hybm'].values
    P0 = grid_ds['P0'].values
    return (A*P0 + B*P_surf(i_column,t))

i_column = 0  # Example column index
t = 2400 # Example time step
pressure_mid = P_mid(i_column,t)
pressure_int = P_int(i_column,t)

lvl_int = np.arange(len(pressure_int))
lvl_mid = np.arange(len(pressure_mid))

plt.scatter(lvl_mid, np.log(pressure_mid), label='P_mid', color='red', marker='+')
plt.scatter(lvl_int, np.log(pressure_int), label='P_int', color='blue', marker='+')
plt.xlabel('Index')
plt.ylabel('Pressure (Pa)')
plt.title('Pressure Profiles')
plt.legend()
plt.grid()
plt.savefig('pressure_profiles.png')



# for i in range (10):
#     t = 2400 + i*1200
#     if t < 10000:
#         t = '0' + str(t)
#     else:
#         t = str(t)
#     state_path = "/work/FAC/FGSE/IDYST/tbeucler/ai4pex/Physics_no_offline/raw_data/train/0001-02/E3SM-MMF.mlexpand.0001-02-01-"+t+".nc"
#     state_ds = xr.open_dataset(state_path)
#     print(state_ds['state_ps'].values[0])

# print(ds)

#print(grid_ds['PS'])
#print(state_ds)


# for var in grid_ds.variables :
#     print(var)

# def P_surf(i_column):
#     return ds['PS'].values[0,i_column]




# # plt.plot(pressure_mid, label='P_mid', color='red')
# # plt.plot(pressure_int, label='P_int', color='blue')
# # plt.xlabel('Index')
# # plt.ylabel('Pressure (Pa)')
# # plt.title('Pressure Profiles')
# # plt.legend()
# # plt.grid()
# # plt.show()


# # Plot hyai, hybi, hyam, hybm
# plt.figure(figsize=(10, 6))

# # # Plot hyai and hybi
# # plt.plot(ds['hyai'].values, range(len(ds['hyai'].values)), label='hyai', marker='o', linestyle='-', color='blue')
# # plt.plot(ds['hybi'].values, range(len(ds['hybi'].values)), label='hybi', marker='o', linestyle='--', color='green')
# print(P_surf(0)-ds['P0'].values)
# # Plot hyam and hybm
# # plt.plot(ds['hyam'].values, range(len(ds['hyam'].values)), label='hyam', marker='x', linestyle='-', color='red')
# # plt.plot(ds['hybm'].values, range(len(ds['hybm'].values)), label='hybm', marker='x', linestyle='--', color='orange')
# # plt.plot(pressure_mid/P_surf(0), range(len(pressure_mid)), label='P_int', color='blue')

# plt.plot(ds['hyai'].values, range(len(ds['hyai'].values)), label='hyai', marker='o', linestyle='-', color='blue')
# plt.plot(ds['hybi'].values, range(len(ds['hybi'].values)), label='hybi', marker='o', linestyle='--', color='green')
# plt.plot(pressure_int/P_surf(0), range(len(pressure_int)), label='P_int/P_surf', color='red')


# plt.ylabel('Index')
# plt.xlabel('Value')
# plt.title('hyai, hybi, hyam, hybm Profiles')
# plt.gca().invert_yaxis()
# plt.legend()
# plt.grid()
# plt.show()