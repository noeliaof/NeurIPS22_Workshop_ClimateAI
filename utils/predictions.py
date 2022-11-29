import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from numpy import ones
import pandas as pd
import xarray as xr

def compute_weighted_score(score, lats_y, lons_x, mean_dims=xr.ALL_DIMS):
    """
    Compute the score with latitude weighting from two xr.DataArrays.
    Args:
        score :  grid-by-grid metric from confusion matrix (e.g., precision, recall, AUC)
        mean_dims: dimensions over which to average score
    Returns:
        m_wg: Latitude weighted root AUC
    """
    
    score = xr.DataArray(score, dims=["lat", "lon"],
                      coords=dict(lat = lats_y, 
                      lon = lons_x))
    weights_lat = np.cos(np.deg2rad(score.lat))
    weights_lat /= weights_lat.mean()
    score_wg = (np.abs(score) * weights_lat).mean(mean_dims)
    return score_wg



def map_metric(m_map, lats_y, lons_x, lead_times, metric):
    """"Function to evaluate spatially the metrics for each lead time
    Args: m_map is a list with arrays for each lead time
          coordinates and lead times
          metric indicates the name of the metric to plot"""
    max_leadtime = len(lead_times)
    x = np.array(m_map)
    xx = xr.DataArray(
                x,
                dims=['lead_time', 'lat', 'lon'],
                coords={'lead_time': lead_times,'lat': lats_y, 'lon': lons_x},
                name = metric)
    p = xx.plot(col='lead_time', col_wrap=max_leadtime, size=3, subplot_kws={'projection': ccrs.PlateCarree()},extend='both')
    
    for ax in p.axes.flat:
        ax.coastlines()
        
        

def get_prediction_scores_leadtime(model, dg_test):
    """Compute predictions for each lead time and calculate the scores
    Args: model used for the training part
          dg_test with the data test for each lead time """
    
    pred_mod = []
    precision_lead_time = []
    recall_lead_time = []
    auc_lead_time = []

    for lead_time in lead_times:

        print(lead_time)

        lt_test = dg_test[lead_time-1]    
        nam_mod = f'ws_xtrm_{model}_lead_time_{lead_time}.h5'
        tmp_file = pathlib.Path (f'tmp/{nam_mod}')
        print(tmp_file)

        if tmp_file.exists :

            print('train the model', model, 'for each lead time: direct forecast')
               # define inputs and outpust
            i_shape = lt_test.data.shape[1:]
            o_shape = lt_test.dy.shape[1:]
            print(o_shape)
            print(i_shape)


            if model == 'resnet':
                mod = build_res_unet(i_shape)
                mod.load_weights(tmp_file)
            else:
                mod = Unet_Inputs('Unet', i_shape, o_shape, input_index=None)
                mod.model.load_weights(tmp_file)
            
            print( 'create predictions')
            pred_lt = create_predictions(mod, lt_test)
            # assess scores
            y_pred_bool = pred_lt > 0.5
            y_pred_bool = y_pred_bool * 1
            y_xtrm = dg_test[lead_time-1].dy.to_numpy().squeeze()
            # score matrix
            precision, recall = eval_confusion_matrix_scores_on_map(y_xtrm[:lt_test.n_samples], y_pred_bool)
            roc_auc = eval_roc_auc_score_on_map(y_xtrm[:lt_test.n_samples], pred_lt)
            # weighted averages metrics
            precision_wg = compute_weighted_score(precision, lats_y, lons_x)
            recall_wg = compute_weighted_score(recall, lats_y, lons_x)
            auc_wg = compute_weighted_score(roc_auc, lats_y, lons_x)


            pred_mod.append(pred_lt)
            precision_lead_time.append(precision_wg)
            recall_lead_time.append(recall_wg)
            auc_lead_time.append(auc_wg)
            

    precision_lead_time = xr.concat(precision_lead_time, 'lead_time')
    recall_lead_time = xr.concat(recall_lead_time, 'lead_time')
    auc_lead_time = xr.concat(auc_lead_time, 'lead_time')
    
    return precision_lead_time, recall_lead_time, auc_lead_time