import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from datetime import datetime
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
gps_1_mask = f.xyz_philam(pos_ECEF, 1)
gps_30_mask = f.xyz_philam(pos_ECEF, 30)

f.plot_groundtrack(gps_1_mask)
f.plot_groundtrack(gps_30_mask)


# Calculate DOP values (Graz)
# VDOP (+vary mask angle)


#HDOP (+vary mask angle)




# Calculate DOP values (Norway)