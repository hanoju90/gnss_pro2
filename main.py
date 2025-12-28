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
f.plot_orbit(pos_ECEF, 'ECEF', [1, 26])


# Visualize satellite coordinates ECSF


# Visualize groundplot
gps_1_mask = f.mask_data(pos_ECEF, 1)
gps_30_mask = f.mask_data(pos_ECEF, 30)

gps1_lam, gps1_phi, gps1_H = f.xyz_LamPhiH(gps_1_mask)
gps30_lam, gps30_phi, gps30_H = f.xyz_LamPhiH(gps_30_mask)

gps1_array = np.column_stack([gps1_lam, gps1_phi, gps1_H])
gps30_array = np.column_stack([gps30_lam, gps30_phi, gps30_H])

f.plot_groundtrack(gps1_array)
f.plot_groundtrack(gps30_array)


# Calculate DOP values (Graz)
# VDOP (+vary mask angle)


graz_xyz = pd.Series([4195411.184, 1156914.919, 4647666.400], index=['X', 'Y', 'Z']) #x,y,z coords of Graz Hbf
graz_lam, graz_phi, graz_H = f.xyz_LamPhiH(graz_xyz)

range = [900, 1200]

pos_ECEF_timemask = f.time_mask(pos_ECEF, range)

dx = pos_ECEF_timemask['X'] - graz_xyz[0]
dy = pos_ECEF_timemask['Y'] - - graz_xyz[1]
dz = pos_ECEF_timemask['Z'] - graz_xyz[2]

rho = np.sqrt(dx**2 + dy**2 + dz**2)

ux = dx/rho
uy = dy/rho
uz = dz/rho

A = np.column_stack([-ux, -uy, -uz, np.ones_like(ux)])

Qx = np.linalg.inv(A.T @ A)

qxx, qyy, qzz, qtt = np.diag(Qx)

PDOP = np.sqrt(qxx + qyy + qzz)

delete_row = np.delete(Qx, 3, 0)
Qxyz = np.delete(delete_row, 3, 1)

R = np.array([[-np.sin(graz_phi)*np.cos(graz_lam), -np.sin(graz_lam), -np.cos(graz_phi)*np.cos(graz_lam)],
              [-np.sin(graz_phi)*np.sin(graz_lam), np.cos(graz_lam), -np.cos(graz_phi)*np.sin(graz_lam)],
              [np.cos(graz_phi), 0, -np.sin(graz_phi)]])


#HDOP (+vary mask angle)




# Calculate DOP values (Norway)