import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from datetime import datetime

from astropy.timeseries.periodograms.lombscargle_multiband.implementations.mle import design_matrix

import function as f


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


# Calculate DOP values
minute_range = [900, 1200]
# Graz
graz_xyz = pd.Series([4195411.184, 1156914.919, 4647666.400], index=['X', 'Y', 'Z']) #x,y,z coords of Graz Hbf
DOP_df = f.calculate_dop_series(pos_ECEF, minute_range, graz_xyz)
print(DOP_df.head())
# Norway


# Results:
# – visualizations and plots (time series number of satellites, skyplots, ...)
# – DOP values and time series