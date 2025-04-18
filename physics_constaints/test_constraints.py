import xarray as xr
import numpy as np

file_path = "example_job_submit_nnwrapper_v4_constrained.eam.h1.0002-12-31-00000.nc"
ds = xr.open_dataset(file_path)
g = 9.81 # to be redefined with the exact fortran code value
cp_air = 1005 # to be redefined with the exact fortran code value
Lv = 334 # to be redefined with the exact fortran code value
Lf = 2264.705 # to be redefined with the exact fortran code value

# Things to edit :
#   fetch all the mising values 
#   check the fortran code for the values of g, cp_air, Lv and Lf
#   check that the sense of the variables is correct (top and bottom of the column)

def T(i_column) : 
    return ds['T'].values[0,:,i_column]

def T_tend(i_column) :
    return ds[''].values[0,:,i_column]

def q(phase, i_column):
    assert phase in ["vapor", "liquid", "ice", "rain", "snow"], "Phase must be vapor, liquid, ice, rain or snow"
    if phase == "vapor":
        return ds['Q'].values[0,:,i_column] 
    elif phase == "liquid":
        return ds['CLDLIQ'].values[0,:,i_column] 
    elif phase == "ice":
        return ds['CLDICE'].values[0,:,i_column] 
    elif phase == "rain":
        raise(ValueError, "q_rain not implemented yet")
        #return ds[''].values[0,:,i_column] 
    elif phase == "snow":
        raise(ValueError, "q_snow not implemented yet")
        #return ds[''].values[0,:,i_column] 

def q_no_precip(i_column):
    return q("vapor", i_column) + q("liquid", i_column) + q("ice", i_column)

def q_total(i_column):
    return q("vapor", i_column) + q("liquid", i_column) + q("ice", i_column) + q("rain", i_column) + q("snow", i_column)

def q_tend(phase, i_column):
    assert phase in ["vapor", "liquid", "ice", "rain", "snow"], "Phase must be vapor, liquid, ice, rain or snow"
    if phase == "vapor":
        return ds['DQ1PHYS'].values[0,:,i_column] 
    elif phase == "liquid":
        return ds['DQ2PHYS'].values[0,:,i_column] 
    elif phase == "ice":
        return ds['DQ3PHYS'].values[0,:,i_column] 
    elif phase == "rain":
        raise(ValueError, "q_rain not implemented yet")
        #return ds[''].values[0,:,i_column] 
    elif phase == "snow":
        raise(ValueError, "q_snow not implemented yet")
        #return ds[''].values[0,:,i_column] 

def q_tend_no_precip(i_column):
    return q_tend("vapor", i_column) + q_tend("liquid", i_column) + q_tend("ice", i_column)

def q_tend_total(i_column): 
    return q_tend("vapor", i_column) + q_tend("liquid", i_column) + q_tend("ice", i_column) + q_tend("rain", i_column) + q_tend("snow", i_column)

def u(i_column):
    return ds['U'].values[0,:,i_column]

def v(i_column):
    return ds['V'].values[0,:,i_column]

def u_tend(i_column):
    return ds[''].values[0,:,i_column]

def v_tend(i_column):
    return ds[''].values[0,:,i_column]

def Precip_snow(i_column):
    return ds[''].values[0,i_column]

def Precip_rain(i_column):
    return ds[''].values[0,i_column]

def Precip_total(i_column):
    return ds[''].values[0,i_column]

def Net_radiation(i_column):
    return ds[''].values[0,i_column]

def P_surf(i_column):
    return ds['PS'].values[0:i_column]

def dP(i_column) :
    A = ds['hyai'].values
    B = ds['hybi'].values
    P0 = ds['P0'].values
    return np.diff(A*P0 + B*P_surf(i_column))

def mass_integration (X,i_column):
    return (X*dP(i_column)/g).sum()

def mass_residual(i_column):
    return mass_integration(q_tend_total(i_column))+ Precip_total(i_column)

def mass_residual_no_precip(i_column):
    return mass_integration(q_tend_no_precip(i_column)) + Precip_total(i_column)

def energy_residual(i_column):
    return (mass_integration(u(i_column)*u_tend(i_column) + v(i_column)*v_tend(i_column) + cp_air*T_tend(i_column) 
                            + (Lv + Lf)*q("vapor", i_column) + Lf*(q("rain", i_column) + q("liquid", i_column)))
                            - Net_radiation(i_column) + Precip_rain(i_column)*Lf)

def energy_residual_no_precip(i_column):
    return (mass_integration(u(i_column)*u_tend(i_column) + v(i_column)*v_tend(i_column) + cp_air*T_tend(i_column) 
                            + (Lv + Lf)*q("vapor", i_column) + Lf*(q("liquid", i_column)))
                            - Net_radiation(i_column) + Precip_rain(i_column)*Lf)