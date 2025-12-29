import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from datetime import datetime

from astropy.timeseries.periodograms.lombscargle_multiband.implementations.mle import design_matrix

import function as f

# Results:
# – visualizations and plots (time series number of satellites, skyplots, ...)
# – DOP values and time series


# Read in data
pos_ECEF = f.read_data("pos_ECEF.txt")
pos_ECSF = f.read_data("pos_ECSF.txt")

# Clean Time and Date columns
pos_ECEF = f.df_cleaning(pos_ECEF)
pos_ECSF = f.df_cleaning(pos_ECSF)


# Visualize ECEF and ECSF for given PRNs
#f.plot_orbit(pos_ECEF, 'ECEF', [1, 26])


# Visualize groundplot
gps_1_mask = f.mask_data(pos_ECEF, 1)
gps_30_mask = f.mask_data(pos_ECEF, 30)

gps1_lam, gps1_phi, gps1_H = f.xyz_LamPhiH(gps_1_mask)
gps30_lam, gps30_phi, gps30_H = f.xyz_LamPhiH(gps_30_mask)

gps1_array = np.column_stack([gps1_lam, gps1_phi, gps1_H])
gps30_array = np.column_stack([gps30_lam, gps30_phi, gps30_H])

#f.plot_groundtrack(gps1_array)
#f.plot_groundtrack(gps30_array)


# Calculate DOP values (Graz)
# VDOP (+vary mask angle)


graz_xyz = pd.Series([4195411.184, 1156914.919, 4647666.400], index=['X', 'Y', 'Z']) #x,y,z coords of Graz Hbf
graz_lam, graz_phi, graz_H = f.xyz_LamPhiH(graz_xyz)

minute_range = [900, 1200]

pos_ECEF_timemask = f.time_mask(pos_ECEF, minute_range)
pos_ECEF_timemask_grouped = pos_ECEF_timemask.groupby('Minutes')

# Rotation matrix for Graz from ECEF to N-E-U system
R_NEU = np.array([[-np.sin(graz_phi)*np.cos(graz_lam), -np.sin(graz_lam), -np.cos(graz_phi)*np.cos(graz_lam)],
              [-np.sin(graz_phi)*np.sin(graz_lam), np.cos(graz_lam), -np.cos(graz_phi)*np.sin(graz_lam)],
              [-np.cos(graz_phi), 0, np.sin(graz_phi)]]) # multiply third row by -1 to get N-E-U instead of N-E-D

# Calculate PDOP for each epoch
for minute, epoch_data in pos_ECEF_timemask_grouped:
    # Build difference vectors
    dx = epoch_data['X'] - graz_xyz[0]
    dy = epoch_data['Y'] - - graz_xyz[1]
    dz = epoch_data['Z'] - graz_xyz[2]
    dX = np.array([dx, dy, dz])

    dX_local_level = R_NEU.T @ dX

    # Calculate zenith angle for visibility mask
    N = dX_local_level[0]
    E = dX_local_level[1]
    U = dX_local_level[2]
    rho_local_level = np.sqrt(N ** 2 + E ** 2 + U ** 2)
    z = np.arccos(U / rho_local_level)
    #print(z)
    elevation = np.pi / 2 - z  # elevation in radians
    elevation_deg = np.degrees(elevation)

    # Exclude satellites by elevation mask
    elevation_mask_deg = 0
    mask = elevation_deg >= elevation_mask_deg
    dx_vis = dx[mask]
    dy_vis = dy[mask]
    dz_vis = dz[mask]

    n_sats = len(dx_vis)

    # Normalise vectors
    rho = np.sqrt(dx_vis**2 + dy_vis**2 + dz_vis**2)
    ux = dx_vis/rho
    uy = dy_vis/rho
    uz = dz_vis/rho

    # Build design matrix
    A = np.column_stack([-ux, -uy, -uz, np.ones_like(ux)])

    Qx = np.linalg.inv(A.T @ A)

    qxx, qyy, qzz, qtt = np.diag(Qx)
    PDOP = np.sqrt(qxx + qyy + qzz)

    # Get Qxyz
    delete_row = np.delete(Qx, 3, 0)
    Qxyz = np.delete(delete_row, 3, 1)

    Qx_local_level = R_NEU.T @ Qxyz @ R_NEU
    #print(Qx_local_level)


    print(minute, n_sats, PDOP)




#HDOP (+vary mask angle)




# Calculate DOP values (Norway)