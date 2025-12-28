import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from pyproj import Transformer


def read_data(pos_filename):
    pos_df = pd.read_csv(
      f"Data/{pos_filename}",
        sep=r"\s+",
        comment="#",
        header=None,
        names=["System", "PRN", "Date", "Time", "X", "Y", "Z"],
    )
    return pos_df


def df_cleaning(pos_df):
    # Convert Date and Time columns to a single column counting the minutes from the measurement start
    pos_df["D"] = pd.to_datetime(pos_df["Date"] + " " + pos_df["Time"], utc=True)
    pos_df = pos_df.drop(columns=["Date", "Time"])
    pos_df["Minutes"] = (pos_df
                           .groupby("PRN")["D"].transform(lambda s: (s - s.min()).dt.total_seconds() / 60))
    return pos_df


def xyz_philam(data, sat_nr):
  mask = data[data['PRN'] == sat_nr]

  transformer = Transformer.from_crs(
    "EPSG:4978",
    "EPSG:4326",
    always_xy=True)

  lat, lon, height = transformer.transform(mask['X'], mask['Y'], mask['Z'])
  philam_array = np.column_stack([lat, lon, height])
  return philam_array


def plot_orbit(pos_df, frame, prn_list):
    # Create earth form for plotting, source https://stackoverflow.com/questions/31768031/plotting-points-on-the-surface-of-a-sphere
    r = 6371000
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0 * pi:100j]
    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi)

    for prn in prn_list:
        pos = pos_df[pos_df['PRN'] == prn]
        x_pos = np.array(pos['X'])
        y_pos = np.array(pos['Y'])
        z_pos = np.array(pos['Z'])
        #f.plot_orbit(x_ecef, y_ecef, z_ecef, 'ECEF', prn)
        plot_title = f"{frame} of PRN {prn}"

        # Visualize satellite coordinates ECEF
        fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        ax = plt.axes(projection='3d')

        # Force same scaling on all 3 axes
        coords = np.concatenate([x_pos, y_pos, z_pos])
        cmin, cmax = coords.min(), coords.max()
        ax.set_xlim(cmin, cmax)
        ax.set_ylim(cmin, cmax)
        ax.set_zlim(cmin, cmax)
        ax.set_box_aspect([1, 1, 1])

        # Earth
        ax.plot_surface(x, y, z, rstride=1, cstride=1, color='c', alpha=0.6, linewidth=0)

        # Orbit
        ax.plot3D(x_pos, y_pos, z_pos, color='k')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(plot_title)
        ax.view_init(elev=20, azim=45)

        plt.show()


def plot_groundtrack(data):
  '''Function to plot the orbit of a grace satellite on a specific background.'''
  plt.figure(figsize=(12, 6))
  ax = plt.axes(projection=ccrs.PlateCarree())

  ax.stock_img()
  ax.coastlines()

  lat = data[:,0]
  lon = data[:,1]
  ax.plot(lat, lon, linewidth=2, color='red', transform=ccrs.Geodetic())
  ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black',
               draw_labels=True, alpha=0.5, linestyle='--')

  plt.title("Groundtrack Plot", fontsize=12)
  plt.show()