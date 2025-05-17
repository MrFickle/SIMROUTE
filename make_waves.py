#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 23:10:59 2020
Code part of SIMROUTE (UPC-BarcelonaTech)
Version: 02 / 02 / 21
@author: manel grifoll (UPC-BarcelonaTech)
"""
import sys
import numpy  as np
from params import prod, date_Ini, date_End, name_Simu, dir_arx
from simroute import tira_lat, tira_lon, Nx, Ny, get_time_tic, print_elapsed_time, compass2cart, arrayRect2Comp
import datetime
from netCDF4 import Dataset 
import scipy.interpolate
import matplotlib.pyplot as plt
import os
os.environ['TCL_LIBRARY'] = r'C:\Users\Koniot\AppData\Local\Programs\Python\Python313\tcl\tcl8.6'

# x -> longtitude
# y -> latitude

# Plot waves after interpolation
plot_waves = True

# Time frame if plotting waves.
t=0

# END OF USER INPUTS   #######################
#creem els arxius diaris aparir del params

# Start timer
tic = get_time_tic()

# Start date as datetime object for validation
d1 = datetime.date(date_Ini[0],date_Ini[1],date_Ini[2])
# End date as datetime object for validation
d2 = datetime.date(date_End[0],date_End[1],date_End[2])
# Convert start date to string
nomw1 = d1.strftime('%Y-%m-%d')
# Convert end date to string
nomw2 = d2.strftime('%Y-%m-%d')

# Wave filename
nomarx = 'Waves_' + name_Simu + '_' + nomw1 + '%' + nomw2 + '.nc'

nc = Dataset(dir_arx + nomarx, 'r')

# Get longtitude, latitude, coordinates, and total points based on product dataset
lon_var = 'lon' if prod in ['XBLKSEA', 'XARTIC'] else 'longitude'
lat_var = 'lat' if prod in ['XBLKSEA', 'XARTIC'] else 'latitude'

lon = nc.variables[lon_var][:]
lat = nc.variables[lat_var][:]

if prod == 'XARTIC':
    # Arctic product has pre-gridded data
    X, Y = lon, lat
    ny, nx = lon.shape
else:
    # Create grid for other products
    X, Y = np.meshgrid(lon, lat)
    nx, ny = len(lon), len(lat)

# Get time dimension if available
ntim = nc.dimensions['time'].size if 'time' in nc.dimensions else None

# Create a meshgrid for longtitude and latitude (longtitude repeated across rows, latitude repeated across columns)
Xnod, Ynod = np.meshgrid(tira_lon, tira_lat)

# Initializations for constructing the wave array

hsi = np.zeros(shape=(ny,nx)) # Significant wave height (instantaneous snapshot)
diri = np.zeros(shape=(ny,nx)) # Wave direction (instantaneous snapshot)
hs_rec = np.zeros(shape=(Ny,Nx,ntim)) # Wave height time series
dir_rec = np.zeros(shape=(Ny,Nx,ntim)) # Wave direction time series

# Since the variables extracted from the NetCDF files behave differently, we need to follow different procedures in these two cases: when there are land points present and when there are no land points at all.
#mnc=Dataset(dir_arx+ARX[0],'r')

# Extract wave height data (first timestep, all lat/lon points)
mhw = nc.variables['VHM0'][0,:,:]

# Check if the data has land masking
if isinstance(mhw.mask, np.ndarray) is False:
    print("No land points")  # Data is entirely ocean
    msk = False  # Flag for "no land"
else:
    print("Land points exist")  # Some values are masked (land)
    msk = True  # Flag for "has land"

#for n in range(Na):
#    nc=Dataset(dir_arx+ARX[n],'r')
#    print(dir_arx+ARX[n])

# Stop timer
toc = get_time_tic()
# Print time passed
print_elapsed_time(tic, toc)

print(f'Unit of time: {ntim}')

# Loop through each time step in the NetCDF file
for t in range(ntim):
    print(t)  # Print current time step for progress tracking

    # 1. Process Wave Height (VHM0) ----------------------------
    hw = nc.variables['VHM0'][t, :, :]  # Get wave height data for time t

    # Handle land masking if needed
    if msk:
        hsi = np.where(~hw.mask, hw.data, np.nan)  # ~ = logical NOT
    else:
        hsi = hw[:]  # No masking needed

    # Interpolate wave height to new grid
    hsg = scipy.interpolate.griddata(
        (X.flatten(), Y.flatten()),  # Original grid points
        hsi.flatten(),  # Wave height values
        (Xnod, Ynod),  # New grid points
        method='linear'  # Linear interpolation
    )
    hs_rec[:, :, t] = hsg[:, :]  # Store interpolated result for time t

    # 2. Process Wave Direction (VMDR) --------------------------
    dirw = nc.variables['VMDR'][t, :, :]  # Get direction data for time t

    # Handle land masking if needed (same as above)
    if msk:
        diri = np.where(~dirw.mask, dirw.data, np.nan)  # ~ = logical NOT
    else:
        diri = dirw[:]  # No masking needed

    # Compass Angles (Meteorological/Oceanographic Convention)
    # 0° = True North, 90° = True East, 180° = True South, 270° = True West (measured clockwise from true North)
    # Used in wave direction (VMDR), wind direction, and navigation
    # Key properties:
    # Circular: 360° = 0° (wraps around)
    # Discontinuity: Linear math (e.g., averaging) fails at 0°/360° boundary, since if you have 2 points at 350° and 10°, the average is 180°, not 0°.

    # Cartesian Angles (Mathematical Convention)
    # 0° = True East, 90° = True North, 180° = True West, 270° = True South (measured counterclockwise from true East)
    # Represented as vectors with x (east-west) and y (north-south) components.
    # Key properties:
    # Works seamlessly with linear algebra and interpolation.

    # Convert direction to Cartesian components for proper interpolation
    dirc = compass2cart(diri)  # Convert angles to 0-360 range if needed
    dir_x = np.cos(np.deg2rad(dirc))  # X-component (east-west)
    dir_y = np.sin(np.deg2rad(dirc))  # Y-component (north-south)

    # Interpolate vector components separately
    dir_xi = scipy.interpolate.griddata(
        (X.flatten(), Y.flatten()),
        dir_x.flatten(),
        (Xnod, Ynod),
        method='linear'
    )
    dir_yi = scipy.interpolate.griddata(
        (X.flatten(), Y.flatten()),
        dir_y.flatten(),
        (Xnod, Ynod),
        method='linear'
    )

    # Convert back to compass angles and store
    dir_rec[:, :, t] = arrayRect2Comp(dir_xi, dir_yi)

print('Interpolation done! Assigning waves at nodes')             
hs=np.zeros(shape=(Nx*Ny,ntim))
hs.fill(np.nan)
#fp=np.copy(hs)
dir=np.copy(hs)

for t in  range(ntim):
    for j in range(Ny):
        for i in range(Nx):
            hs[i+j*Nx,t]=hs_rec[j,i,t];
 #           fp[i+j*Nx,t]=fp_rec[j,i,t];
            dir[i+j*Nx,t]=dir_rec[j,i,t];

#Nan extraction in Dir due to interpolation 
for i in  range(Nx*Ny):
    if  np.isnan(hs[i,0])==False and np.isnan(dir[i,0])==True:
                hs[i,:]=np.nan
                print('algun nan fa la punyeta ',i,j)

if plot_waves is False:
    print("Delete intermediate variables.")       
    del dir_xi
    del dir_yi
    del dir_rec
#    del fp_rec
    del hs_rec
print ("Checking nans in waves fields.")
[nn,tt]=hs.shape
n=0
for i in range(nn):
    val_ini=np.isnan(hs[i,0])
    for t in range(tt):
        if val_ini !=np.isnan(hs[i,t]):
            hs[i,:]=np.nan
            dir[i,:]=np.nan
#            fp[i,:]=np.nan 
            n=n+1
            print(i,t,n)
            break
if n !=0:
    print('Find nans and eliminated :',n )



                
print("Done. Saving...")
arxi='in/'+name_Simu+'_wInt.npz'            
np.savez_compressed(arxi,hs,dir)


toc()

if plot_waves is True:
    fig=plt.figure()
    t=10
    axes=fig.add_axes([0.1,0.1,0.8,0.8])
    axes.set_ylabel('Lat (º)')
    axes.set_xlabel('Lon (º)')
    axes.set_title('Significant wave hight (in m) in time : '.format(t))
    ima=axes.pcolor(Xnod,Ynod,hs_rec[:,:,t],vmin=0,vmax=np.nanmax(hs_rec))
    plt.colorbar(ima)    
    plt.show()

