
import sklearn
assert sklearn.__version__ >= '0.20'

from sklearn.impute import SimpleImputer
#from sklearn.pipeline import Pipelines
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# TensorFlow ≥2.0 is required
import tensorflow_addons as tfa
import tensorflow as tf
assert tf.__version__ >= '2.0'

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model

from keras.models import Sequential
from tensorflow.keras.layers import Dropout, BatchNormalization, Reshape
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, Input, MaxPooling2D, Flatten, MaxPool2D, MaxPool3D, UpSampling2D
from tensorflow.keras.layers import Conv2DTranspose, Flatten, Reshape, Cropping2D, Embedding, BatchNormalization,ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU, Activation, Input, add, multiply
from tensorflow.keras.layers import ConvLSTM1D
from tensorflow.keras.layers import concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Lambda
import tensorflow.keras.backend as K

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
import re
import pdb
import logging
from tqdm import tqdm


dask.config.set({'array.slicing.split_large_chunks': False})

# To make this notebook's output stable across runs
np.random.seed(42)

# Dotenv
#from dotenv import dotenv_values
# Custom utils
from utils.utils_data import *
from utils.utils_ml import *
from utils.utils_plot import *
from utils.utils_unet import *
from utils.utils_resnet import *
from utils.DNN_models import *
from utils.networks import *
from utils.datagenerators import *
from utils.utils_predictions import * 


conf = yaml.safe_load(open("config.yaml"))
conf['PATH_ERA5']
# open data 
ws = xr.open_mfdataset('data/ws.nc')
ds = xr.open_mfdataset('data/ds_all.nc')
mjo = xr.open_mfdataset('data/mjo.nc')


mjo.copy()
lats_y = ws.lat
lons_x = ws.lon
# adding dimensions to the temporal index MJO
mjo_dim = mjo.expand_dims({'lat': lats_y,'lon':lons_x})
mjo_dim = mjo_dim.transpose('time','lat','lon')



#ws = ws.ws
ws['time'] = pd.DatetimeIndex(ws.time.dt.date)


# adding more indices
df_aao = pd.read_csv('/storage/workspaces/giub_hydro/hydro/data/NOAA_index/norm.daily.aao.cdas.z700.19790101_current.csv')
df_ao = pd.read_csv('/storage/workspaces/giub_hydro/hydro/data/NOAA_index/norm.daily.ao.cdas.z1000.19500101_current.csv')
df_nao = pd.read_csv('/storage/workspaces/giub_hydro/hydro/data/NOAA_index/norm.daily.nao.cdas.z500.19500101_current.csv')
df_pna = pd.read_csv('/storage/workspaces/giub_hydro/hydro/data/NOAA_index/norm.daily.pna.cdas.z500.19500101_current.csv')
aao = prepare_telec_indices(df_aao, 'aao_index_cdas', 'AAO', lats_y, lons_x, conf['DATE_START'], conf['DATE_END'])
ao = prepare_telec_indices(df_ao, 'ao_index_cdas', 'AO', lats_y, lons_x, conf['DATE_START'], conf['DATE_END'])
nao = prepare_telec_indices(df_nao, 'nao_index_cdas', 'NAO', lats_y, lons_x, conf['DATE_START'], conf['DATE_END'])
pna = prepare_telec_indices(df_pna, 'pna_index_cdas', 'PNA', lats_y, lons_x, conf['DATE_START'], conf['DATE_END'])


ws.load()


# target variable
ws_weekly = ws.ws.resample(time="1W").mean(dim="time")



def weekly_standard_anomalies(ds, groupby = 'week'):
    """
    Output standardized weekly anomalies, climatology, and std.
    """

    climatology_mean = ds.groupby('time' + '.' + groupby).mean('time')
    climatology_std = ds.groupby('time' + '.' + groupby).std('time')

    stand_anomalies = xr.apply_ufunc(
                                lambda x, m, s: (x - m) / s,
                                ds.groupby('time' + '.' + groupby),
                                climatology_mean,
                                climatology_std,
                            )
    anomalies = xr.apply_ufunc(
                                lambda x, m: (x - m),
                                ds.groupby('time' + '.' + groupby),
                                climatology_mean,
                            )
    return stand_anomalies, anomalies


# ### Defining extremes
# 1. Define extremes based on percentile
# wind_weekly < 1th 
qt = 0.1
qq = xr.DataArray(ws_weekly).quantile(qt, dim='time') 
low_ws = xr.DataArray(ws_weekly < qq)
low_ws= low_ws*1



# ### Definition based on standarized anomalies


ws_anom_sd, ws_anom = weekly_standard_anomalies(ws_weekly)

# Define wind droughts based on standarised anomalies, let's start wih -1 to have more data
st_ex = -1
ws_ext = xr.where(ws_anom_sd < st_ex, 1, 0)


# test all data
# Aggregate weekly
ds_tot = xr.merge([ds, mjo_dim, aao, ao, nao, pna],compat='no_conflicts')
dstot_weekly = ds_tot.resample(time='1W').mean()

dic_all = {
     'z': [1000, 850, 500, 300],              #let's use more geopotential levels
     'msl': None,
     't2m': None,
     'u': [300,925],
     'v': [300,925],
     'RMM1': None, 
     'RMM2': None,
     'amplitude': None,
     'phase': None,
     'AAO': None,
     'AO':None,
     'NAO': None,
     'PNA': None
    }



# Split into training and test, then I will use DataGenerator class to get the validation
dstot_train = dstot_weekly.sel(time=slice('{}-01-01'.format(conf['YY_TRAIN'][0]),
                             '{}-12-31'.format(conf['YY_TRAIN'][1])))

dstot_train['time'] = pd.DatetimeIndex(dstot_train.time.dt.date)
dstot_test = dstot_weekly.sel(time=slice('{}-01-01'.format(conf['YY_TEST'][0]),
                            '{}-12-31'.format(conf['YY_TEST'][1])))

dstot_test['time'] = pd.DatetimeIndex(dstot_test.time.dt.date)


dy_train = ws_weekly.sel(time=slice(f"{conf['YY_TRAIN'][0]}", f"{conf['YY_TRAIN'][1]}"))
dy_test = ws_weekly.sel(time=slice(f"{conf['YY_TEST'][0]}", f"{conf['YY_TEST'][1]}"))
# Next, TAKE EXTREMES!
# First based on percentile
dyex_train = low_ws.sel(time=slice(f"{conf['YY_TRAIN'][0]}", f"{conf['YY_TRAIN'][1]}"))
dyex_test = low_ws.sel(time=slice(f"{conf['YY_TEST'][0]}", f"{conf['YY_TEST'][1]}"))
# based on anomalies
dyex2_train = ws_ext.sel(time=slice(f"{conf['YY_TRAIN'][0]}", f"{conf['YY_TRAIN'][1]}"))
dyex2_test = ws_ext.sel(time=slice(f"{conf['YY_TEST'][0]}", f"{conf['YY_TEST'][1]}"))



# #### Datagenerator 
# Direct forecast: here we train the model separately for each lead time 
BATCH_SIZE=32
max_leadtime = 8
lead_times = np.arange(1, max_leadtime + 1)
print(lead_times)


# model hyperparameters
output_scaling = 1
output_crop = None
EPOCHS = 50
#Early stopping
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                            restore_best_weights=True)

# Default training options
opt_training = {'epochs': EPOCHS,
                'callbacks': [callback]}



# compute weights
# Compute weights for the weighted binary crossentropy
weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(low_ws.values),
    y=low_ws.values.flatten()
)

print('Weights for the weighted binary crossentropy:')
print(f'Classes: {np.unique(low_ws.values)}, weights: {weights}')

# Create loss function for the extremes
xtrm_loss = weighted_binary_cross_entropy(
    weights={0: weights[0].astype('float32'), 1: weights[1].astype('float32')})


models = {
          'Unet': {'model': 'Unet', 'run': True,
                   'opt_model': {'output_scaling': output_scaling, 'output_crop': output_crop, 'unet_depth': 4, 'dropout_rate': 0.1, 'unet_use_upsample': False, 'bottleneck': False,  'for_extremes' : True},
                    'opt_optimizer': {'lr_method': 'Constant','lr':.00001}},
          'UnetConv': {'model': 'UnetConv', 'run': True,
                   'opt_model': {'output_scaling': output_scaling, 'output_crop': output_crop, 'unet_depth': 4, 'unet_use_upsample': False, 'bottleneck': False,  'for_extremes' : True},
                   'opt_optimizer': {'lr_method': 'Constant', 'lr':.00001}},
          'resnet': {'model': 'resnet', 'run': True,
                    'opt_model': {'out_channels': 1,'for_extremes' : True},
                    'opt_optimizer': {'lr_method': 'Constant','lr':.00008}}}



loss = xtrm_loss
metric = 'accuracy'


dg_train = []
dg_test = []
dg_valid = []
plot_his = False 

m_lt = []
hist_lt = []


for lead_time in lead_times:
    print(lead_time)
    lt_train = DataGenerator(dstot_train.sel(time=slice(f"{conf['YY_TRAIN'][0]}", f"{conf['YY_VALID']}")), 
                                             dyex_train.sel(time=slice(f"{conf['YY_TRAIN'][0]}", f"{conf['YY_VALID']}")),
                                             dic_all, lead_time=lead_time, batch_size=BATCH_SIZE, load=True)

    lt_valid = DataGenerator(dstot_train.sel(time=slice(f"{conf['YY_VALID']+1}", f"{conf['YY_TRAIN'][1]}")), 
                                             dyex_train.sel(time=slice(f"{conf['YY_VALID']+1}", f"{conf['YY_TRAIN'][1]}")),
                                             dic_all, lead_time=lead_time,  batch_size=BATCH_SIZE, mean=lt_train.mean, std=lt_train.std, load=True)
        #test
    lt_test = DataGenerator(dstot_test, dyex_test, dic_all, lead_time=lead_time,
                                            batch_size=BATCH_SIZE, mean=lt_train.mean, std=lt_train.std, shuffle=False, load=True)
    dg_train.append(lt_train)
    dg_valid.append(lt_valid)
    dg_test.append(lt_test)
    
    
    for m_id in models:


         # Clear session and set tf seed
        keras.backend.clear_session()
        tf.random.set_seed(42)

        if not models[m_id]['run']:
            continue
        # Extract model name and options
        model = models[m_id]['model']
        opt_model_i = models[m_id]['opt_model']
        opt_optimizer_i = models[m_id]['opt_optimizer']
        opt_model_new = opt_model_i.copy()
        opt_model_new.update(opt_model_i)
        opt_optimizer_new = opt_optimizer_i.copy()
        opt_optimizer_new.update(opt_optimizer_i)
        optimizer =  initiate_optimizer(lt_train, BATCH_SIZE, **opt_optimizer_new)
        print('train the model', model, 'for each lead time: direct forecast')
        print(opt_optimizer_new)
       # define inputs and outpust
        i_shape = lt_train.data.shape[1:]
        o_shape = lt_train.dy.shape[1:]
        print(o_shape)
        print(i_shape)
        

            
        if model == 'resnet':
            
            mod = build_res_unet(i_shape, **opt_model_new)
        else:
            mod = Unet_Inputs('Unet', i_shape, o_shape, input_index=None, **opt_model_new)


        mod.compile(loss=loss, optimizer=optimizer,  metrics=metric)
        his_mod = mod.fit(lt_train, validation_data=lt_valid, **opt_training)

        if plot_his:
            pd.DataFrame(his_mod.history)[['loss','val_loss']].plot(figsize=(8,6), grid=True)
            plt.title('model' + model + 'loss for lead time:' + str(lead_time))

        # save models
        nam_mod = f'ws_xtrm_{model}_lead_time_{lead_time}.h5'
        if model == 'resnet':
            mod.save_weights(f'tmp/{nam_mod}')
        else:
            mod.model.save_weights(f'tmp/{nam_mod}')
   
        m_lt.append(mod)
        hist_lt.append(his_mod)


