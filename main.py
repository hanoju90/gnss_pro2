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
graz_xyz = pd.Series([4195411.184, 1156914.919, 4647666.400], index=['X', 'Y', 'Z']) # Coordinates of Graz Hbf
#DOP_graz_df = f.calculate_dop_series(pos_ECEF, minute_range, graz_xyz)
#print(DOP_graz_df.head())

# Norway
bodo_xyz = pd.Series([2392289, 613884, 5860822], index=['X', 'Y', 'Z']) # Coordinates of Bodo, Norway
#DOP_bodo_df = f.calculate_dop_series(pos_ECEF, minute_range, bodo_xyz)
#print(DOP_bodo_df.head())




# Results:
# – visualizations and plots (time series number of satellites, skyplots, ...)

elevation_angles = [0, 5, 10, 15]


# Plot Number of Satellites and PDOP for different elevation angles in Graz and Bodo (individually)
for elevation_angle in elevation_angles:
    f.plot_nr_sats(pos_ECEF, minute_range, graz_xyz, elevation_angle, 'Graz')
    f.plot_nr_sats(pos_ECEF, minute_range, bodo_xyz, elevation_angle, 'Bodo')
    f.plot_dop_timeseries(pos_ECEF, minute_range, graz_xyz, elevation_angle, 'Graz', 'PDOP')
    f.plot_dop_timeseries(pos_ECEF, minute_range, bodo_xyz, elevation_angle, 'Bodo', 'PDOP')



# Plot Number of Satellites for different elevation angles in Graz (comparison)
f.plot_nr_sats_comparison(pos_ECEF, minute_range, graz_xyz, elevation_angles, 'Graz')
f.plot_dop_timeseries_comparison(pos_ECEF, minute_range, graz_xyz, elevation_angles, 'Graz', 'PDOP')
f.plot_dop_timeseries_comparison(pos_ECEF, minute_range, graz_xyz, elevation_angles, 'Graz', 'HDOP')
f.plot_dop_timeseries_comparison(pos_ECEF, minute_range, graz_xyz, elevation_angles, 'Graz', 'VDOP')

# Plot Number of Satellites for different elevation angles in Bodo (comparison)
f.plot_nr_sats_comparison(pos_ECEF, minute_range, bodo_xyz, elevation_angles, 'Bodo')
f.plot_dop_timeseries_comparison(pos_ECEF, minute_range, bodo_xyz, elevation_angles, 'Bodo', 'PDOP')
f.plot_dop_timeseries_comparison(pos_ECEF, minute_range, bodo_xyz, elevation_angles, 'Bodo', 'HDOP')
f.plot_dop_timeseries_comparison(pos_ECEF, minute_range, bodo_xyz, elevation_angles, 'Bodo', 'VDOP')



# TODO: Visualize skyplots
# TODO: Exclude individual satellites and analyse DOP timeseries
