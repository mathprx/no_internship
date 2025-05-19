import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from soft_interpolation import soft_interpolation

g = 9.81

file_path = "Grid/example_job_submit_nnwrapper_v4_constrained.eam.h0.0002-12.nc"
ds = xr.open_dataset(file_path)

def P_surf(i_column):
    return ds['PS'].values[0,i_column]

def P_int(i_column) :
    A = ds['hyai'].values
    B = ds['hybi'].values
    P0 = ds['P0'].values
    return A*P0 + B*P_surf(i_column)

def T(i_column) : 
    return ds['T'].values[0,:,i_column]

p_int = P_int(0)
segments = [(p_int[i], p_int[i+1]) for i in range(len(p_int)-1)]
means = T(0)*g

interpolation = soft_interpolation(segments=segments, means=means)
interpolation.solve()
interpolation.visualize()
