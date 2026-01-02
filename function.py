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


def dms_to_rad(d,m,s):
  """
  function to convert degree, minutes and seconds into radians
  Args:
    d: degree
    m: minutes
    s: seconds

  Returns: radians

  """
  d_float = d + float(m)/60 + float(s)/3600
  rad = np.radians(d_float)
  return rad

def mask_data(data, sat_nr):
  mask = data[data['PRN'] == sat_nr]
  return mask

def xyz_LamPhiH(data):

  transformer = Transformer.from_crs(
    "EPSG:4978",
    "EPSG:4326",
    always_xy=True)

  lon, lat, height = transformer.transform(data['X'], data['Y'], data['Z'])
  #LamPhiH_array = np.column_stack([lon, lat, height])
  return lon, lat, height


def time_mask(df, range):
  start = range[0]
  stop = range[1]
  df = df.loc[(df["Minutes"] >= start) & (df["Minutes"] <= stop)]
  return df


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

  lon = data[:,0]
  lat = data[:,1]
  ax.plot(lon, lat, linewidth=2, color='red', transform=ccrs.Geodetic())
  ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black',
               draw_labels=True, alpha=0.5, linestyle='--')

  plt.title("Groundtrack Plot", fontsize=12)
  plt.show()


def create_rotation_matrix(pos_xyz):
    '''Create rotation matrix to go from ECEF to N-E-U at a specific position'''
    pos_lam, pos_phi, pos_H = xyz_LamPhiH(pos_xyz)
    R_NEU = np.array([[-np.sin(pos_phi) * np.cos(pos_lam), -np.sin(pos_lam), -np.cos(pos_phi) * np.cos(pos_lam)],
                       [-np.sin(pos_phi) * np.sin(pos_lam), np.cos(pos_lam), -np.cos(pos_phi) * np.sin(pos_lam)],
                       [-np.cos(pos_phi), 0,
                        np.sin(pos_phi)]])  # multiply third row by -1 to get N-E-U instead of N-E-D
    return R_NEU

def calculate_dop_series(pos_ECEF, minute_range, pos_xyz):
    '''Function to calculate DOP values and nr of visible satellites at each epoch'''
    pos_ECEF_timemask = time_mask(pos_ECEF, minute_range)
    pos_ECEF_timemask_grouped = pos_ECEF_timemask.groupby('Minutes')

    # Rotation matrix for Graz to go from ECEF to N-E-U
    R_NEU = create_rotation_matrix(pos_xyz)

    results = []

    for minute, epoch_data in pos_ECEF_timemask_grouped:
        # Build difference vectors
        dx = epoch_data['X'] - pos_xyz['X']
        dy = epoch_data['Y'] - pos_xyz['Y']
        dz = epoch_data['Z'] - pos_xyz['Z']
        dX = np.array([dx, dy, dz])

        dX_local_level = R_NEU.T @ dX

        # Calculate zenith angle for visibility mask
        rho_local_level = np.linalg.norm(dX_local_level,2)
        z = np.arccos(dX_local_level[2] / rho_local_level)
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
        rho = np.sqrt(dx_vis ** 2 + dy_vis ** 2 + dz_vis ** 2)
        ux = dx_vis / rho
        uy = dy_vis / rho
        uz = dz_vis / rho

        # Build design matrix
        A = np.column_stack([-ux, -uy, -uz, np.ones_like(ux)])

        Qx = np.linalg.inv(A.T @ A)

        qxx, qyy, qzz, qtt = np.diag(Qx)
        PDOP = np.sqrt(qxx + qyy + qzz)

        # Get Qxyz
        delete_row = np.delete(Qx, 3, 0)
        Qxyz = np.delete(delete_row, 3, 1)

        Qx_local_level = R_NEU.T @ Qxyz @ R_NEU
        # print(Qx_local_level)

        # Calculate HDOP and VDOP
        qnn, qee, qdd = np.diag(Qx_local_level)
        HDOP = np.sqrt(qnn + qee)
        VDOP = np.sqrt(qdd)

        results.append({
            "Minutes": minute,
            "n_sats": n_sats,
            "PDOP": PDOP,
            "HDOP": HDOP,
            "VDOP": VDOP, })

    DOP_df = pd.DataFrame(results).sort_values("Minutes")
    return DOP_df