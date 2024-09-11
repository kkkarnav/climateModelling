import os
import glob
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm

def read_netcdf_tmax(file_path):

    with nc.Dataset(file_path, 'r') as ds:
        tmax = ds.variables['tmax'][:]
        lats = ds.variables['lat'][:]
        lons = ds.variables['lon'][:]
        time = ds.variables['time'][:]
    
    # Assuming tmax is 4D (time, level, lat, lon)
    tmax_yearly = np.max(tmax, axis=(0, 1))
    
    return tmax_yearly, lats, lons, time

def create_tmax_plot(tmax, lats, lons, year, vmin, vmax):
    
    
    
    tmax_celsius = tmax - 273.15
    
    plt.figure(figsize=(12, 8))
    
    
    contour = plt.contourf(lons, lats, tmax_celsius, cmap='coolwarm', levels=20, vmin=vmin, vmax=vmax)
    
    
    cbar = plt.colorbar(contour, label='Maximum Temperature (°C)')
    
    plt.title(f'Maximum Temperature - Year {year}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
    
    
    plt.xticks(np.arange(np.min(lons), np.max(lons), 15))  
    plt.yticks(np.arange(np.min(lats), np.max(lats), 15))  
    
    
    plt.savefig(f'./images/ncep_visuals/globals/temp_tmax_{year}.png')
    plt.close()

def create_tmax_gif(netcdf_dir, output_gif):


    netcdf_files = sorted(glob.glob(os.path.join(netcdf_dir, '*.nc')))
    
    images = []
    all_tmax = []  
    
    for file in netcdf_files:
        year = os.path.basename(file).split('.')[3]  
        tmax, lats, lons, _ = read_netcdf_tmax(file)
        all_tmax.append(tmax)  
    
    # Determine global min and max for consistent color scale
    all_tmax = np.concatenate(all_tmax, axis=0)  
    vmin = np.min(all_tmax) - 273.15  
    vmax = np.max(all_tmax) - 273.15  
    
    
    for file in tqdm(netcdf_files):
        year = os.path.basename(file).split('.')[3]
        tmax, lats, lons, _ = read_netcdf_tmax(file)
        create_tmax_plot(tmax, lats, lons, year, vmin, vmax)
        images.append(imageio.imread(f'./images/ncep_visuals/globals/temp_tmax_{year}.png'))
        os.remove(f'./images/ncep_visuals/globals/temp_tmax_{year}.png') 
    
    
    imageio.mimsave(output_gif, images, duration=1)  # 1 second per frame



# India-specific functions

def read_netcdf_tmax_india(file_path):
    with nc.Dataset(file_path, 'r') as ds:
        tmax = ds.variables['tmax'][:]
        lats = ds.variables['lat'][:]
        lons = ds.variables['lon'][:]
    
    
    lat_mask = (lats >= 8) & (lats <= 38)
    lon_mask = (lons >= 68) & (lons <= 98)
    
    tmax_india = tmax[:, :, lat_mask, :][:, :, :, lon_mask]
    lats_india = lats[lat_mask]
    lons_india = lons[lon_mask]
    
    
    tmax_yearly = np.max(tmax_india, axis=(0, 1))
    
    return tmax_yearly, lats_india, lons_india



def create_tmax_plot_india(tmax, lats, lons, year, vmin, vmax):
    tmax_celsius = tmax - 273.15
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a mesh grid for pcolormesh
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)
    
    
    im = ax.pcolormesh(lon_mesh, lat_mesh, tmax_celsius, cmap='coolwarm', vmin=vmin, vmax=vmax)
    
    
    cbar = plt.colorbar(im, label='Maximum Temperature (°C)')
    
    
    ax.set_xticks(np.arange(68, 98.1, 2.5))
    ax.set_yticks(np.arange(8, 38.1, 2.5))
    ax.grid(True, which='both', color='black', linewidth=0.5)
    
    plt.title(f'Maximum Temperature in India - Year {year}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    
    plt.savefig(f'./images/ncep_visuals/india/temp_tmax_india_{year}.png')
    plt.close()

def create_tmax_gif_india(netcdf_dir, output_gif):
    netcdf_files = sorted(glob.glob(os.path.join(netcdf_dir, '*.nc')))
    
    images = []
    all_tmax = []
    
    for file in netcdf_files:
        year = os.path.basename(file).split('.')[3]
        tmax, lats, lons = read_netcdf_tmax_india(file)
        all_tmax.append(tmax)
    
    all_tmax = np.concatenate(all_tmax, axis=0)
    vmin = np.min(all_tmax) - 273.15
    vmax = np.max(all_tmax) - 273.15
    
    for file in tqdm(netcdf_files):
        year = os.path.basename(file).split('.')[3]
        tmax, lats, lons = read_netcdf_tmax_india(file)
        create_tmax_plot_india(tmax, lats, lons, year, vmin, vmax)
        images.append(imageio.imread(f'./images/ncep_visuals/india/temp_tmax_india_{year}.png'))
        os.remove(f'./images/ncep_visuals/india/temp_tmax_india_{year}.png')
    
    imageio.mimsave(output_gif, images, duration=1)


if __name__ == "__main__":
    netcdf_dir = './dataset/NCEP/NCEP-Global-tmax-4x-daily/'
    output_gif = './images/ncep_visuals/globals/tmax_changes.gif'
    output_gif_india = './images/ncep_visuals/india/tmax_changes_india.gif'
    
    create_tmax_gif(netcdf_dir, output_gif)
    create_tmax_gif_india(netcdf_dir, output_gif_india)