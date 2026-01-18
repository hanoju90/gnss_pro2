import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from itertools import cycle
import datetime
from skyfield.api import Topos, load, wgs84
from skyfield.sgp4lib import EarthSatellite


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

def xyz_to_lamphih(data):
  X = data['X']
  Y = data['Y']
  Z = data['Z']

  a = 6378137.00000 #m
  b = 6356752.31425 #m

  e_strich_sq = (a**2 - b**2) / b**2
  e_sq = (a**2 - b**2)/ a**2
  c = (a**2)/b
  p = np.sqrt(X**2 + Y**2)
  theta = np.atan2(Z * a, p * b)

  phi = np.atan2(Z + e_strich_sq * b * np.sin(theta)**3, p - e_sq * a * np.cos(theta)**3)
  V = np.sqrt(1 + e_strich_sq * np.cos(phi)**2)
  lam = np.atan2(Y, X)
  phi_deg = phi * 180/np.pi
  lam_deg = lam * 180/np.pi
  h = (p/np.cos(phi)) - (c/V)

  return lam_deg, phi_deg, h

def time_mask(df, range):
  start = range[0]
  stop = range[1]
  df = df.loc[(df["Minutes"] >= start) & (df["Minutes"] <= stop)]
  return df

def plot_orbit(pos_df, frame, prn_list, compare=0):
    # Create earth form for plotting, source https://stackoverflow.com/questions/31768031/plotting-points-on-the-surface-of-a-sphere
    r = 6371000
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0 * pi:100j]
    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi)

    if compare == 1:
        ax_comp = plt.axes(projection='3d')
        coords_all = np.concatenate([
            pos_df['X'].to_numpy(),
            pos_df['Y'].to_numpy(),
            pos_df['Z'].to_numpy()
        ])
        cmin_all, cmax_all = coords_all.min(), coords_all.max()

    for prn in prn_list:
        pos = pos_df[pos_df['PRN'] == prn]
        x_pos = pos['X']
        y_pos = pos['Y']
        z_pos = pos['Z']

        if compare == 0:
            plot_title = f"{frame} of PRN {prn}"
            ax = plt.axes(projection='3d')

            # Force same scaling on all 3 axes
            coords = np.concatenate([x_pos, y_pos, z_pos])
            cmin, cmax = coords.min(), coords.max()
            ax.set_xlim(cmin, cmax)
            ax.set_ylim(cmin, cmax)
            ax.set_zlim(cmin, cmax)
            ax.set_box_aspect([1, 1, 1])

            # Earth
            u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
            ax.plot_wireframe(r * np.cos(u) * np.sin(v), r * np.sin(u) * np.sin(v), r * np.cos(v), color="black",
                              alpha=0.4, lw=1, zorder=0)
            ax.plot_surface(x, y, z, rstride=1, cstride=1, color='lightgrey', alpha=0.4, linewidth=0)

            # Orbit
            ax.plot(x_pos, y_pos, z_pos, color='tab:red', linewidth=2, alpha=1)

            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.set_zlabel('z [m]')
            ax.set_title(plot_title)
            # ax.view_init(elev=30, azim=120)
            plt.savefig(f"Results/Orbit_{prn}_{frame}.png")
            plt.close()
        elif compare == 1:
            # Orbit
            color = np.random.rand(3, )
            ax_comp.plot(x_pos, y_pos, z_pos, color=color, linewidth=2, label=f"PRN {prn}", alpha=1)

    if compare == 1:
        plot_title = f"Comparison of all Satellite Orbits in {frame}"
        # Force same scaling on all axes
        ax_comp.set_xlim(cmin_all, cmax_all)
        ax_comp.set_ylim(cmin_all, cmax_all)
        ax_comp.set_zlim(cmin_all, cmax_all)
        ax_comp.set_box_aspect([1, 1, 1])

        # Earth
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        ax_comp.plot_wireframe(r * np.cos(u) * np.sin(v), r * np.sin(u) * np.sin(v), r * np.cos(v), color="black",
                               alpha=0.4, lw=1, zorder=0)
        ax_comp.plot_surface(x, y, z, rstride=1, cstride=1, color='lightgrey', alpha=0.4, linewidth=0)

        ax_comp.set_xlabel('x [m]')
        ax_comp.set_ylabel('y [m]')
        ax_comp.set_zlabel('z [m]')
        ax_comp.set_title(plot_title)
        ax_comp.legend(bbox_to_anchor=(1.12, 0.5), loc="center left", borderaxespad=0)
        # ax.view_init(elev=30, azim=120)
        # plt.show()
        plt.savefig(f"Results/Orbits_comparison_{frame}.png", bbox_inches="tight")
        plt.close()





def plot_groundtrack(data, prn_nr):
  '''Function to plot the orbit of a grace satellite on a specific background.'''
  plt.figure(figsize=(12, 6))
  ax = plt.axes(projection=ccrs.PlateCarree())
  #ax = plt.axes()

  ax.stock_img()
  ax.coastlines()

  lon = data[:,0]
  lat = data[:,1]
  ax.plot(lon, lat, linewidth=2, color='red', transform=ccrs.Geodetic())
  ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black',
               draw_labels=True, alpha=0.5, linestyle='--')

  plt.title(f"Groundtrack Plot PRN {prn_nr}", fontsize=12)
  plt.savefig(f"Results/Groundplot_{prn_nr}")
  plt.close()


def create_rotation_matrix(pos_xyz):
    '''Create rotation matrix to go from ECEF to N-E-U at a specific position'''
    # pos_lam, pos_phi, pos_H = xyz_LamPhiH(pos_xyz)
    #transformer = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)
    pos_lam_deg, pos_phi_deg, pos_h = xyz_to_lamphih(pos_xyz)
    pos_lam = np.radians(pos_lam_deg)
    pos_phi = np.radians(pos_phi_deg)
    R_NEU = np.array([[-np.sin(pos_phi) * np.cos(pos_lam), -np.sin(pos_lam), np.cos(pos_phi) * -np.cos(pos_lam)],
                       [-np.sin(pos_phi) * np.sin(pos_lam), np.cos(pos_lam), np.cos(pos_phi) * -np.sin(pos_lam)],
                       [np.cos(pos_phi), 0,
                        -np.sin(pos_phi)]])
    return R_NEU

def find_visible_sats(epoch_data, pos_xyz, R_NEU, elevation_angle):
    dx = epoch_data['X'] - pos_xyz['X']
    dy = epoch_data['Y'] - pos_xyz['Y']
    dz = epoch_data['Z'] - pos_xyz['Z']
    dX = np.array([dx, dy, dz])
    # Rotation matrix for Graz to go from ECEF to N-E-U
    dX_local_level = R_NEU.T @ dX  # Rotate difference vector between receiver and satellite to local level frame
    # Calculate zenith angle for visibility mask
    dX_norm_local_level = np.linalg.norm(dX_local_level, axis=0)  # Distance between receiver and satellite
    z = np.arccos(-dX_local_level[2, :] / dX_norm_local_level)  # Divide Z/Up-component by distance
    elevation = np.pi / 2 - z  # elevation in radians

    # Exclude satellites by elevation mask
    mask = elevation >= np.radians(elevation_angle)
    # All angles greater than the defined mask angle
    dx_vis = dx[mask]
    dy_vis = dy[mask]
    dz_vis = dz[mask]
    prns_vis = epoch_data['PRN'].to_numpy()[mask]
    return dx_vis, dy_vis, dz_vis, prns_vis



def calculate_dop_series(pos_ECEF, minute_range, pos_xyz, elevation_angle):
    '''Function to calculate DOP values and nr of visible satellites at each epoch'''
    pos_ECEF_timemask = time_mask(pos_ECEF, minute_range)
    pos_ECEF_timemask_grouped = pos_ECEF_timemask.groupby('Minutes')

    # Rotation matrix for Graz to go from ECEF to N-E-U
    R_NEU = create_rotation_matrix(pos_xyz)

    results = []

    for minute, epoch_data in pos_ECEF_timemask_grouped:
        dx_vis, dy_vis, dz_vis, prn_vis = find_visible_sats(epoch_data, pos_xyz, R_NEU, elevation_angle)
        n_sats = len(dx_vis)

        if n_sats < 4:
            results.append({
            "Minutes": minute,
            "n_sats": n_sats,
            "PDOP": np.nan,
            "HDOP": np.nan,
            "VDOP": np.nan})
            continue

        # Normalise vectors
        rho = np.sqrt(dx_vis ** 2 + dy_vis ** 2 + dz_vis ** 2) # Distance between receiver and satellite
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
        qnn, qee, quu = np.diag(Qx_local_level)
        HDOP = np.sqrt(qnn + qee)
        VDOP = np.sqrt(quu)

        results.append({
            "Minutes": minute,
            "n_sats": n_sats,
            "PDOP": PDOP,
            "HDOP": HDOP,
            "VDOP": VDOP})

    DOP_df = pd.DataFrame(results).sort_values("Minutes")
    return DOP_df

def plot_nr_sats(pos_ECEF, minute_range, pos_xyz, elevation_angle, place_name):
    DOP_df = calculate_dop_series(pos_ECEF, minute_range, pos_xyz, elevation_angle)
    fig = plt.figure()
    plt.plot(DOP_df['Minutes'], DOP_df['n_sats'], color='blue')
    plt.xlabel("Time [min]")
    plt.ylabel("Number of visible Satellites")
    plt.suptitle(f"{place_name}: Number of Satellites", fontsize=16)
    plt.title(f'Elevation Angle: {elevation_angle}', fontsize=12)
    plt.grid()
    plt.show()
    #plt.savefig(f"Results/{place_name}_nr_sats_{elevation_angle}.png")
    #plt.close()

def plot_dop_timeseries(pos_ECEF, minute_range, pos_xyz, elevation_angle, place_name, dop_type):
    DOP_df = calculate_dop_series(pos_ECEF, minute_range, pos_xyz, elevation_angle)
    plt.figure()
    plt.plot(DOP_df['Minutes'], DOP_df[f'{dop_type}'], color='blue')
    plt.xlabel("Time [min]")
    plt.ylabel(f"{dop_type} Values")
    plt.suptitle(f"{place_name}: {dop_type}", fontsize=16)
    plt.title(f'Elevation Angle {elevation_angle}', fontsize=12)
    plt.grid()
    plt.show()
    #plt.savefig(f"Results/{place_name}_{dop_type}_{elevation_angle}.png")
    #plt.close()


def plot_nr_sats_comparison(pos_ECEF, minute_range, pos_xyz, elevation_angles, place_name):
    plt.figure()
    cmap = plt.get_cmap("tab10")
    color_cycle = cycle(cmap.colors)
    for elevation_angle in elevation_angles:
        DOP_df = calculate_dop_series(pos_ECEF, minute_range, pos_xyz, elevation_angle)
        plt.plot(DOP_df['Minutes'], DOP_df['n_sats'],
                 color=next(color_cycle), label=f"{elevation_angle}°")
    plt.xlabel("Time [min]")
    plt.ylabel("Number of visible Satellites")
    plt.suptitle(f"Number of Satellites with different Elevation Angles", fontsize=16)
    plt.title(f'{place_name}', fontsize=12)
    plt.grid()
    plt.legend()
    plt.show()
    #plt.savefig(f'Results/{place_name}_nr_sats_comparison')
    #plt.close()

def plot_dop_timeseries_comparison(pos_ECEF, minute_range, pos_xyz, elevation_angles, place_name, dop_type):
    plt.figure()
    cmap = plt.get_cmap("tab10")
    color_cycle = cycle(cmap.colors)
    for elevation_angle in elevation_angles:
        DOP_df = calculate_dop_series(pos_ECEF, minute_range, pos_xyz, elevation_angle)
        plt.plot(DOP_df['Minutes'], DOP_df[f'{dop_type}'],
                 color=next(color_cycle), label=f"{elevation_angle}°")
    plt.xlabel("Time [min]")
    plt.ylabel(f"{dop_type} Values")
    plt.suptitle(f"{dop_type} with different Elevation Angles", fontsize=16)
    plt.title(f'{place_name}', fontsize=12)
    plt.grid()
    plt.legend()
    plt.show()
    #plt.savefig(f'Results/{place_name}_{dop_type}_comparison')
    #plt.close()

def exclude_sats(data, exclude_sat_list=[]):
  removed_data = data
  for i in exclude_sat_list:
    removed_data = removed_data.drop(removed_data[removed_data['PRN'] == i].index)
  return removed_data


def plot_skyplots(pos_ecef, obs_xyz, elevation_angle, place_name):
  R = create_rotation_matrix(obs_xyz)

  dx_vis, dy_vis, dz_vis, prns_vis = find_visible_sats(pos_ecef, obs_xyz, R, elevation_angle)

  dX_local = np.vstack([dx_vis, dy_vis, dz_vis])
  dX_ll = R.T @ dX_local

  N, E, D = dX_ll

  dX_norm_ll = np.linalg.norm(dX_ll, axis=0)
  z = np.arccos(-dX_ll[2, :] / dX_norm_ll)
  elevation = np.pi / 2 - z

  elevation_deg = np.degrees(elevation)
  elevation_deg = np.clip(elevation_deg, 0.0, 90.0)

  az = np.arctan2(E, N)
  az_deg = (np.degrees(az) + 360) % 360

  theta = np.radians(az_deg)
  r = 90 - elevation_deg

  fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))
  fig.subplots_adjust(right=0.70)
  for prn in np.unique(prns_vis):
    mask = prns_vis == prn
    ax.scatter(theta[mask], r[mask], s=0.5, label=f'PRN {prn}')

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rticks([0, 30, 60, 80])
    ax.set_yticklabels(['90°', '60°', '30°', '10°'])
    ax.set_rlabel_position(0)

    ax.set_thetagrids([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330], labels=['0°', '30°', '60°', '90°', '120°', '150°', '180°', '210°', '240°', '270°', '300°', '330°'])
    ax.set_rlim(0, 90)
    ax.grid(True)
    ax.legend(loc='center right', bbox_to_anchor=(1.45, 0.5), fontsize=9)
    plt.suptitle(f'Skyplot {place_name}', fontsize=16, y=0.94)
    plt.title(f'Elevation Mask: {elevation_angle}°', fontsize=12, x=0.65, pad=15)

  plt.show()

