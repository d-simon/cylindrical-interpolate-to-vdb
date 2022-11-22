import pandas as pd
import numpy as np
import h5py
import itertools
from scipy.interpolate import griddata


def pol_to_cart(angle_pol, x_pol, z_pol):
    '''
    Parameters:
    angle_pol: angle of plane
    x_pol : horizontal amplitude
    z_pol : vertical coordinate

    Return:
    x_cart, y_cart, z_cart: cartesion coordinates
    '''

    z = x_pol * np.exp(1j * angle_pol)
    x_cart, y_cart = z.real, z.imag
    z_cart = z_pol

    return x_cart, y_cart, z_cart

def ext_interpol(df_data, new_index):
    df_data = df_data.reindex(index=new_index)
    df_data = df_data.sort_index()
    df_data = df_data.interpolate()
    return df_data


filename = 'results_52.h5'
frame_string = '000566'
theta = np.array(h5py.File(filename, 'r')['data']['var3d']['theta'][frame_string])
temp = np.array(h5py.File(filename, 'r')['data']['var3d']['temperature'][frame_string])
angle = np.array(h5py.File(filename, 'r')['data']['var3d']['temperature']['coord3'])
pol_x = np.array(h5py.File(filename, 'r')['data']['var3d']['temperature']['coord1'])
pol_z = np.array(h5py.File(filename, 'r')['data']['var3d']['temperature']['coord2'])

df_temp = pd.DataFrame(index=angle, data=temp.reshape((80, -1)))
df_theta = pd.DataFrame(index=angle, data=theta.reshape((80, -1)))

# create new index vector --> add new interpolation points (at the moment double)
angle_ext = np.append(angle, angle + np.diff(np.append(angle, 2 * np.pi)) / 2)

df_temp = ext_interpol(df_temp, angle_ext)
df_theta = ext_interpol(df_theta, angle_ext)
# df_pressure = df_theta * df_temp

# Create vector with anglexz coordinates
coords_pol = list(itertools.product(angle_ext, pol_x, pol_z))
v_angle, v_x, v_z = np.array(list(map(list, zip(*coords_pol))))

# Create Vector with xyz target coordinates for all anglexz coordinates
x_cart, y_cart, z_cart = pol_to_cart(v_angle, v_x, v_z)
coords_cart = np.array([x_cart, y_cart, z_cart]).T

# Define Evaluation grid
grid_x, grid_y, grid_z = np.mgrid[-750:750:300j, -750:750:300j, 0:900:225j]
# Interpolate NaNs in 3d
result_temp = griddata(coords_cart, df_temp.values.ravel(), (grid_x, grid_y, grid_z), method='nearest')
result_theta = griddata(coords_cart, df_theta.values.ravel(), (grid_x, grid_y, grid_z), method='nearest')

results_pressure = result_theta*result_temp
