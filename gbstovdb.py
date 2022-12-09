import pandas as pd
import numpy as np
import h5py
import itertools
import sys
from scipy.interpolate import griddata
import pathlib
import pyopenvdb as vdb
import time

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


def convert_and_interpolate_to_cartesian(temp, theta, angle, pol_x, pol_z, xdim_len, ydim_len, zdim_len, xdim_min, xdim_max, ydim_min, ydim_max, zdim_min, zdim_max):
    df_temp = pd.DataFrame(index=angle, data=temp.reshape((80, -1)))
    df_theta = pd.DataFrame(index=angle, data=theta.reshape((80, -1)))

    # create new index vector --> add new interpolation points (at the moment double)
    angle_ext = np.append(angle, angle + np.diff(np.append(angle, 2 * np.pi)) / 2)

    # df_temp = ext_interpol(df_temp, angle_ext)
    # df_theta = ext_interpol(df_theta, angle_ext)
    # df_pressure = df_theta * df_temp

    # Create vector with anglexz coordinates
    coords_pol = list(itertools.product(angle, pol_x, pol_z))
    v_angle, v_x, v_z = np.array(list(map(list, zip(*coords_pol))))

    # Create Vector with xyz target coordinates for all anglexz coordinates
    x_cart, y_cart, z_cart = pol_to_cart(v_angle, v_x, v_z)
    coords_cart = np.array([x_cart, y_cart, z_cart]).T

    # Define Evaluation grid
    grid_x, grid_y, grid_z = np.mgrid[xdim_min:xdim_max:xdim_len*1j, ydim_min:ydim_max:ydim_len*1j, zdim_min:zdim_max:zdim_len*1j]
    # Interpolate NaNs in 3d
    result_temp = griddata(coords_cart, df_temp.values.ravel(), (grid_x, grid_y, grid_z), method='nearest', fill_value=0.0)
    result_theta = griddata(coords_cart, df_theta.values.ravel(), (grid_x, grid_y, grid_z), method='nearest', fill_value=0.0)

    return result_temp, result_theta

filename = sys.argv[1]
output_dirname = sys.argv[2]
# pathlib.Path(filename).stem
# specific_frame_str = sys.argv[2]

start_time = time.time()
# make directory
pathlib.Path('output/' + output_dirname  + '/').mkdir(parents=True, exist_ok=True)

lod = 1 # level of detail

f = h5py.File(filename, 'r')
data = f['data']
var3d = data['var3d']

angle = np.array(var3d['temperature']['coord3'])
pol_x = np.array(var3d['temperature']['coord1'])
pol_z = np.array(var3d['temperature']['coord2'])

r_size = var3d['theta']['coord1'].shape[0]
z_size = var3d['theta']['coord2'].shape[0]
slices = var3d['theta']['coord3'].shape[0]

rmin = var3d['theta']['coord1'][0]
rmax = var3d['theta']['coord1'][r_size-1]
zmin = var3d['theta']['coord2'][0]
zmax = var3d['theta']['coord2'][z_size-1]

print('rmin', rmin)
print('rmax', rmax)
print('zmin', zmin)
print('zmax', zmax)

# "boundary" dimensions
xdim_min = -rmax
xdim_max = rmax
ydim_min = xdim_min
ydim_max = xdim_max
zdim_min = zmin
zdim_max = zmax

# calculate aspect ratio with the diff between coordiantes
coordinate_aspect_ratio = (var3d['theta']['coord1'][r_size-1] - var3d['theta']['coord1'][r_size-2]) / (var3d['theta']['coord2'][z_size-1] - var3d['theta']['coord2'][z_size-2])

# get dimensions of grid / cube
xdim_len = 2 * (r_size + round((r_size / (rmax - rmin))*rmin))
ydim_len = xdim_len
zdim_len = round(coordinate_aspect_ratio * (z_size + round((z_size / (zmax - zmin))*rmin)))

# adjust to level-of-detail
xdim_len = round(xdim_len / lod)
ydim_len = round(ydim_len / lod)
zdim_len = round(zdim_len / lod)

print('xdim_len', xdim_len)
print('ydim_len', ydim_len)
print('zdim_len', zdim_len)

# calculate voxelsize
realsize = (xdim_max - xdim_min) / 1000 # mm to meters
voxelsize = realsize / xdim_len
print('realsize', realsize)
print('voxelsize', voxelsize)

keys = var3d['temperature'].keys()
frames = [x for x in keys if all(y not in x for y in ['coord'])]
# frames = ['000833', '000834']
print('frames', frames)

print(f"Starting execution after {(time.time() - start_time)} seconds")

for frame_str in frames:

    theta = np.array(var3d['theta'][frame_str])
    temp = np.array(var3d['temperature'][frame_str])

    # do the deed
    result_temp, result_theta = convert_and_interpolate_to_cartesian(
        temp, theta, angle, pol_x, pol_z, xdim_len, ydim_len, zdim_len, xdim_min, xdim_max, ydim_min, ydim_max, zdim_min, zdim_max)

    density = np.exp(result_temp)
    temperature = np.exp(result_theta)
    pressure = density * temperature

    # vdb
    grid_temperature = vdb.FloatGrid()
    grid_pressure = vdb.FloatGrid()
    grid_density = vdb.FloatGrid()
    grid_density.transform = vdb.createLinearTransform(voxelSize=voxelsize)
    grid_pressure.transform = vdb.createLinearTransform(voxelSize=voxelsize)
    grid_temperature.transform = vdb.createLinearTransform(voxelSize=voxelsize)

    # calculate grid offset for vdb
    grid_offset = (
        round(xdim_min/(xdim_max-xdim_min)*xdim_len),
        round(ydim_min/(ydim_max-ydim_min)*ydim_len),
        round(zdim_min/(zdim_max-zdim_min)*zdim_len),
    )

    # VDB
    grid_temperature.copyFromArray(temperature, grid_offset)
    grid_density.copyFromArray(density, grid_offset)
    grid_pressure.copyFromArray(pressure, grid_offset)

    # add parameters
    grid_temperature.name = 'temperature'
    grid_density.name = 'density'
    grid_pressure.name = 'pressure'

    # write to file
    vdb.write('output/' + output_dirname  + '/' + frame_str + '.vdb', grids=[grid_temperature, grid_density, grid_pressure])

    print(f"Writing frame {frame_str} after {(time.time() - start_time) / 60} minutes")

print("---")
print(f"Finished after total {(time.time() - start_time) / 60} minutes")