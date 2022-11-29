
# Common imports
import os
import glob
import numpy as np
from numpy import ones
import pandas as pd
#import geopandas as gpd
import xarray as xr
import dask
import datetime
import math
import pathlib
import hashlib
import yaml
#dask.config.set({'array.slicing.split_large_chunks': False})

# Custom utils
from utils.utils_data import *
from utils.utils_ml import *
from utils.utils_plot import *
from utils.utils_unet import *
from utils.utils_resnet import *

conf = yaml.safe_load(open("config.yaml"))
conf['PATH_ERA5']



# open data predictors
ws = get_nc_data(conf['PATH_ERA5'] + '/Wind/*nc', conf['DATE_START'], conf['DATE_END'], conf['LONS_eur'], conf['LATS_eur'])
# Extract coordinates for precip
lats_y = ws.lat.to_numpy()
lons_x = ws.lon.to_numpy()
ws = ws.ws
ws['time'] = pd.DatetimeIndex(ws.time.dt.date)

# load predictors
l_paths = ['/geopotential/','/MSLP/','/t2m/','/temperature/', '/U/', '/V/', '/TCW/']
v_vars = ['z','msl','t2m', 't', 'u', 'v', 'tcw']
list_vars = load_data(v_vars, l_paths, conf['G'], conf['PATH_ERA5'], conf['DATE_START'], conf['DATE_END'], conf['LONS_eur'], conf['LATS_eur'], conf['LEVELS'])
datasets = list_vars

# Read indices
mjo = xr.open_mfdataset('/storage/workspaces/giub_hydro/hydro/data/Indices/*.nc')
mjo.load()
mjo['time'] = pd.to_datetime(mjo['time'], format='%Y%m%d')
mjo = mjo.sel(time=slice(conf['DATE_START'], conf['DATE_END']))


# Check if all have the same latitude order
datasets_W = []
for idat in range(0, len(datasets)):

    # Invert lat axis if needed
    if datasets[idat].lat[0].values < datasets[idat].lat[1].values:
        print('change lat order', idat)
        datasets[idat] = datasets[idat].reindex(lat=list(reversed(datasets[idat].lat)))

        dat_W =datasets[idat].resample(time='1W').mean()
        datasets_W.append(dat_W)
        
        
ds = xr.merge(datasets)
ds_weekly = xr.merge(datasets_W)


# Aggregate them weekly
ds.to_netcdf('data/ds_1959-2021.nc')
#ds_weekly.to_netcdf('data/ds_weekly_all.nc')
ws.to_netcdf('data/ws_1959-2021.nc')
mjo.to_netcdf('data/mjo_1974-2021.nc')
