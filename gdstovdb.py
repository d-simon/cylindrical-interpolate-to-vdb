import h5py
import pyopenvdb as vdb
import numpy as np
from sys import argv
import math


# arguments
filename = argv[1]
frame_start = int(argv[2]) # 517
frame_end = int(argv[3]) # 525
pad_to_figures = 6 #000517

# level of detail (1 = full, 2 = half, 4 = forth etc.)
lod = 1

# get data
f = h5py.File(filename,'r')
data = f['data']
var3d = data['var3d']


for frame in range(frame_start, frame_end+1):
    frame_str = str(frame).rjust(pad_to_figures, '0')

    frame_theta = var3d['theta'][frame_str]
    frame_temp = var3d['temperature'][frame_str]

    r_size = var3d['theta']['coord1'].shape[0]
    z_size = var3d['theta']['coord2'].shape[0]
    slices = var3d['theta']['coord3'].shape[0]

    rmin = var3d['theta']['coord1'][0]
    rmax = var3d['theta']['coord1'][r_size-1]
    zmin = var3d['theta']['coord2'][0]
    zmax = var3d['theta']['coord2'][z_size-1]

    print('frame', frame_str)
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

    # create grids
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
    print('grid_offset', grid_offset)

    # create arrays
    array_temperature = np.zeros((xdim_len, ydim_len, zdim_len))
    array_pressure = np.zeros((xdim_len, ydim_len, zdim_len))
    array_density = np.zeros((xdim_len, ydim_len, zdim_len))

    for ix in range(xdim_len):
        print('progress:' + str(math.floor((ix)/(xdim_len)*100*10)/10) + '%')
        for iy in range(ydim_len):
            for iz in range(zdim_len):
                # compute x, y, z from grid ix, iy, iz
                x = xdim_min + (xdim_max - xdim_min) * (ix / xdim_len)
                y = ydim_min + (ydim_max - ydim_min) * (iy / ydim_len)
                z = zdim_min + (zdim_max - zdim_min) * (iz / zdim_len)

                # compute r
                r = np.sqrt(x**2 + y**2)

                if (rmin <= r and r < rmax):
                    # get rz index
                    index_r = math.floor(((r - rmin) / (rmax - rmin)) * r_size)
                    index_z = math.floor(((z - zmin) / (zmax - zmin)) * z_size)

                    # get angle
                    angle = np.arctan2(x, y)

                    # naive impelmentation: get nearest slice
                    nearest_slice = math.floor(((angle + np.pi) / (2 * np.pi)) * slices) - 1

                    theta = f['data']['var3d']['theta'][frame_str][nearest_slice,index_r,index_z]
                    temp = f['data']['var3d']['temperature'][frame_str][nearest_slice,index_r,index_z]

                    density = np.exp(theta)
                    temperature = np.exp(temp)
                    pressure = density * temperature

                    array_temperature[ix, iy, iz] = temperature
                    array_pressure[ix, iy, iz] = pressure
                    array_density[ix, iy, iz] = density

    grid_temperature.copyFromArray(array_temperature, grid_offset)
    grid_density.copyFromArray(array_density, grid_offset)
    grid_pressure.copyFromArray(array_pressure, grid_offset)

    # add parameters
    grid_temperature.name = 'temperature'
    grid_density.name = 'density'
    grid_pressure.name = 'pressure'


    # write to file
    vdb.write('output/' + frame_str + '-' + str(lod) + '.vdb', grids=[grid_temperature, grid_density, grid_pressure])