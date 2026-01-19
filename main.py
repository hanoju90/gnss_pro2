import pandas as pd
import numpy as np
import function as f

print("====== GNSS Lab 02: Satellite constellations and calculation of DOP-values ======")

# Define variables
minute_range = [900, 1200]
# ECEF position of Graz
graz_xyz = pd.Series([4195411.184, 1156914.919, 4647666.400], index=['X', 'Y', 'Z']) # Coordinates of Graz Hbf
# ECEF position of Norway
bodo_xyz = pd.Series([2392289, 613884, 5860822], index=['X', 'Y', 'Z']) # Coordinates of Bodo, Norway
# Elevation angles for elevation masks
elevation_angles = [0, 5, 10, 15]
# List of satellites to be excluded
exclude_sat_list = [1, 2, 3, 4, 7, 30]

# Read in data
pos_ECEF = f.read_data("pos_ECEF.txt")
pos_ECSF = f.read_data("pos_ECSF.txt")

# Clean Time and Date columns
pos_ECEF = f.df_cleaning(pos_ECEF)
pos_ECSF = f.df_cleaning(pos_ECSF)

# Satellite position df with satellites excluded
pos_ECEF_excluded_sats = f.exclude_sats(pos_ECEF, exclude_sat_list)


print("Data import and cleaning done...")

# ========== RESULTS: Plots of satellite orbits, Number of visible satellites and DOP timeseries ==========
print("Generating plots of satellite orbits...")

# Visualize ECEF and ECSF satellite orbits for selected PRNs
prn_list = [1, 3, 5, 30]
f.plot_orbit(pos_ECEF, 'ECEF', prn_list)
f.plot_orbit(pos_ECSF, 'ECSF', prn_list)

# Visualize ECEF and ECSF satellite orbits: comparison of all satellites
prn_list_all_ECEF = pos_ECEF['PRN'].unique().tolist()
f.plot_orbit(pos_ECEF, 'ECEF', prn_list_all_ECEF, 1)
prn_list_all_ECSF = pos_ECSF['PRN'].unique().tolist()
f.plot_orbit(pos_ECSF, 'ECSF', prn_list_all_ECSF, 1)

print("Generating groundplots...")

# Visualize groundplots
gps_1_mask = f.mask_data(pos_ECEF, 1)
gps_30_mask = f.mask_data(pos_ECEF, 30)
gps1_lam, gps1_phi, gps1_H = f.xyz_to_lamphih(gps_1_mask)
gps30_lam, gps30_phi, gps30_H = f.xyz_to_lamphih(gps_30_mask)

gps1_array = np.column_stack([gps1_lam, gps1_phi, gps1_H])
gps30_array = np.column_stack([gps30_lam, gps30_phi, gps30_H])

f.plot_groundtrack(gps1_array, 1)
f.plot_groundtrack(gps30_array, 30)

# Generate statistics of PDOP values and Number of Satellites
print("Calculating statistics for PDOP values...")
f.calculate_pdop_stats(pos_ECEF, minute_range, graz_xyz, elevation_angles, 'Graz-Austria')
f.calculate_pdop_stats(pos_ECEF_excluded_sats, minute_range, graz_xyz, elevation_angles, 'Graz-Austria', exclude_sat_list)

f.calculate_pdop_stats(pos_ECEF, minute_range, bodo_xyz, elevation_angles, 'Bodo-Norway')
f.calculate_pdop_stats(pos_ECEF_excluded_sats, minute_range, bodo_xyz, elevation_angles, 'Bodo-Norway', exclude_sat_list)


# Plot Number of Satellites and PDOP for different elevation angles in Graz and Bodo (individually)
print("Generating plots for nr. of visible satellites and DOP timeseries...")

for elevation_angle in elevation_angles:
    # Plot number of visible satellites
    f.plot_nr_sats(pos_ECEF, minute_range, graz_xyz, elevation_angle, 'Graz-Austria')
    f.plot_nr_sats(pos_ECEF, minute_range, bodo_xyz, elevation_angle, 'Bodo-Norway')

    # Plot DOP timeseries
    f.plot_dop_timeseries(pos_ECEF, minute_range, graz_xyz, elevation_angle, 'Graz-Austria', 'PDOP')
    f.plot_dop_timeseries(pos_ECEF, minute_range, bodo_xyz, elevation_angle, 'Bodo-Norway', 'PDOP')
    # DOP timeseries with excluded sats
    f.plot_dop_timeseries(pos_ECEF_excluded_sats, minute_range, graz_xyz, elevation_angle, 'Graz-Austria', 'PDOP', exclude_sat_list)
    f.plot_dop_timeseries(pos_ECEF_excluded_sats, minute_range, bodo_xyz, elevation_angle, 'Bodo-Norway', 'PDOP', exclude_sat_list)

    # Plot skyplots
    f.plot_skyplots(pos_ECEF, minute_range, graz_xyz, elevation_angle, 'Graz-Austria')
    f.plot_skyplots(pos_ECEF, minute_range, bodo_xyz, elevation_angle, 'Bodo-Norway')

    # Skyplots with excluded sats
    f.plot_skyplots(pos_ECEF_excluded_sats, minute_range, graz_xyz, elevation_angle, 'Graz-Austria', exclude_sat_list)
    f.plot_skyplots(pos_ECEF_excluded_sats, minute_range, bodo_xyz, elevation_angle, 'Bodo-Norway', exclude_sat_list)


# Plot Number of Satellites for different elevation angles in Graz (comparison)
f.plot_nr_sats_comparison(pos_ECEF, minute_range, graz_xyz, elevation_angles, 'Graz-Austria')

# Plot PDOP, HDOP and VDOP timeseries for Graz (comparison)
f.plot_dop_timeseries_comparison(pos_ECEF, minute_range, graz_xyz, elevation_angles, 'Graz-Austria', 'PDOP')
f.plot_dop_timeseries_comparison(pos_ECEF, minute_range, graz_xyz, elevation_angles, 'Graz-Austria', 'HDOP')
f.plot_dop_timeseries_comparison(pos_ECEF, minute_range, graz_xyz, elevation_angles, 'Graz-Austria', 'VDOP')
# with excluded sats
f.plot_dop_timeseries_comparison(pos_ECEF_excluded_sats, minute_range, graz_xyz, elevation_angles, 'Graz-Austria',
                                 'PDOP', exclude_sat_list)


# Plot Number of Satellites for different elevation angles in Bodo (comparison)
f.plot_nr_sats_comparison(pos_ECEF, minute_range, bodo_xyz, elevation_angles, 'Bodo-Norway')

# Plot PDOP, HDOP and VDOP timeseries for Bodo (comparison)
f.plot_dop_timeseries_comparison(pos_ECEF, minute_range, bodo_xyz, elevation_angles, 'Bodo-Norway', 'PDOP')
f.plot_dop_timeseries_comparison(pos_ECEF, minute_range, bodo_xyz, elevation_angles, 'Bodo-Norway', 'HDOP')
f.plot_dop_timeseries_comparison(pos_ECEF, minute_range, bodo_xyz, elevation_angles, 'Bodo-Norway', 'VDOP')
# with excluded sats
f.plot_dop_timeseries_comparison(pos_ECEF_excluded_sats, minute_range, bodo_xyz, elevation_angles, 'Bodo-Norway',
                                 'PDOP', exclude_sat_list)