import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import dask
import datetime
from dotenv import dotenv_values
from scipy.stats import pearsonr


def rename_dimensions_variables(ds):
    """Rename dimensions and attributes of the given dataset to homogenize data."""
    if 'latitude' in ds.dims:
        ds = ds.rename({'latitude': 'lat'})
    if 'longitude' in ds.dims:
        ds = ds.rename({'longitude': 'lon'})

    return ds


def temporal_slice(ds, start, end):
    """Slice along the temporal dimension."""
    ds = ds.sel(time=slice(start, end))

    if 'time_bnds' in ds.variables:
        ds = ds.drop('time_bnds')

    return ds


def spatial_slice(ds, lon_bnds, lat_bnds):
    """Slice along the spatial dimension."""
    if lon_bnds != None:
        ds = ds.sel(lon=slice(min(lon_bnds), max(lon_bnds)))

    if lat_bnds != None:
        if ds.lat[0].values < ds.lat[1].values:
            ds = ds.sel(lat=slice(min(lat_bnds), max(lat_bnds)))
        else:
            ds = ds.sel(lat=slice(max(lat_bnds), min(lat_bnds)))

    return ds


def get_nc_data(files, start, end, lon_bnds=None, lat_bnds=None):
    """Extract netCDF data for the given file(s) pattern/path."""
    print('Extracting data for the period {} - {}'.format(start, end))
    ds = xr.open_mfdataset(files, combine='by_coords')
    ds = rename_dimensions_variables(ds)
    # convert longitudes-this applies to global grids
    if np.all(ds.lon) >= 0:
        ds=ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
        ds = ds.roll(lon=int(len(ds['lon']) / 2),roll_coords=True)
        
    ds = temporal_slice(ds, start, end)
    ds = spatial_slice(ds, lon_bnds, lat_bnds)

    return ds


def get_era5_data(files, start, end, lon_bnds=None, lat_bnds=None):
    """Extract ERA5 data for the given file(s) pattern/path."""
    
    return get_nc_data(files, start, end, lon_bnds, lat_bnds)


def precip_exceedance(precip, qt=0.95):
    """Create exceedances of precipitation

    Arguments:
    precip -- the precipitation dataframe
    qt -- the desired quantile
    """
    precip_qt = precip.copy()

    for key, ts in precip.iteritems():
        if key in ['date', 'year', 'month', 'day']:
            continue
        precip_qt[key] = ts > ts.quantile(qt)

    return precip_qt


def precip_exceedance_xarray(precip, qt=0.95):
    """Create exceedances of precipitation

    Arguments:
    precip -- xarray with precipitation 
    qt -- the desired quantile
    """
    qq = xr.DataArray(precip).quantile(qt, dim='time') 
    out = xr.DataArray(precip > qq)
    out = out*1

    return out


def get_Y_sets(dat, YY_TRAIN, YY_VAL, YY_TEST):
    """ Prepare the targe Y for train, validation and test """
    # Prepare the target Y
    # Split the prec into the same 
    Y_train = dat.sel(time=slice('{}-01-01'.format(YY_TRAIN[0]),
                             '{}-12-31'.format(YY_VAL)))
    Y_val = dat.sel(time=slice('{}-01-01'.format(YY_TRAIN[1]),
                             '{}-12-31'.format(YY_TRAIN[1])))
    Y_test = dat.sel(time=slice('{}-01-01'.format(YY_TEST[0]),
                            '{}-12-31'.format(YY_TEST[1])))
    Y_train_input = np.array(Y_train)
    Y_val_input = np.array(Y_val)
    Y_test_input = np.array(Y_test)

    return Y_train_input, Y_val_input, Y_test_input




def load_data(i_vars, i_paths, G, PATH_ERA5, DATE_START, DATE_END, LONS, LATS, LEVELS):
    """Load the data
       Args: 
       Var: variables
       PATH_ERA5: path to the era5 datasets
       DATE_START: starting date
       DATE_END: end date
       LONS: longitudes
       LATS: latitudes"""

    
    l_vars = []
    for iv in range(0,len(i_vars)):
        
        vv = get_era5_data(PATH_ERA5 + i_paths[iv] +'*nc', DATE_START, DATE_END, LONS, LATS)
        #if i_vars[iv] != 'tpcw' and i_vars[iv] != 't2m' and i_vars[iv] != 'msl':
        if 'level' in list(vv.coords): 
            print("select level")
            #print(vv.level)
            # levels
            lev = np.array(vv.level)
            l=[x for x in lev if x in LEVELS]
            #print(l)
            vv = vv.sel(level=l)
            
        if i_vars[iv] == 'z':
            vv.z.values = vv.z.values/G
        #elif i_vars[iv] == 'rh':
        #    vv = vv.sel(level=LEVELS)
            
        vv['time'] = pd.DatetimeIndex(vv.time.dt.date)
    
        l_vars.append(vv)

    
    
    return l_vars



def mjo_correlations(mjo, ws, index):
    """Calculate the correlations between MJO and W10"""
    cor_m = np.zeros(ws.shape[1:])
    
    for i in range(0,len(ws.lat)):
        for j in range(0, len(ws.lon)):
            corval,_ = pearsonr(mjo[index], ws[:,i,j]) 
            cor_m[i,j] = corval
            
    xx = xr.DataArray(cor_m, dims=["lat", "lon"],
                  coords=dict(lat = ws.lat, 
            lon = ws.lon))
            
    return xx



def prepare_telec_indices(df, vname, sname,lats_y, lons_x, date_start, date_end, expand =True):
    """Function to convert the dataframe into the array
       Args: df with day, month, year and index value
             vname: name of the index
             sname: name to use in the xarray
             lats_y, lons_x: coordinates
             date_starts, date_end: dates to use """
    # check missing values
    df.iloc[df[(df.isnull().sum(axis=1) >= 1)].index]
    df = df.fillna(df.mean())
    df['time'] = pd.to_datetime(df[["year", "month", "day"]])
    df['time'] = pd.to_datetime(df.time)
   # t_index =  xr.Dataset(df[vname], dims='time',
   #               coords=dict(time = df['time']))
    t_index = xr.Dataset(
                {sname:(["time"], df[vname])},
                coords={"time":df['time']},
            )
    
  #  t_index.name = sname
    
    if expand:
        t_index_dim = t_index.expand_dims({'lat': lats_y,'lon':lons_x})
        t_index_dim = t_index_dim.transpose('time','lat','lon')
    t_index_dim = t_index_dim.sel(time=slice(date_start, date_end))
        
    return(t_index_dim)