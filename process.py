#'../../load_forecasting/Lastprognose/heikendorf/GWH/Heikendorf_Hammerstiel.xlsx'

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
from tsmoothie.smoother import KalmanSmoother
from dateutil.relativedelta import relativedelta


def correct_frequency(dataframe):
    data = dataframe.copy()
    data = data[~data.index.duplicated(keep='first')]
    correct_freq = []
    for item in data.index.year.unique():
        first_index = data[data.index.year == item].index[0]
        next_year = first_index + relativedelta(years=1)
        while(first_index < next_year):
            correct_freq.append(first_index)
            first_index = first_index + timedelta(minutes=15)
            
    correct_frq = pd.DataFrame({ 'Date': pd.Series(correct_freq, dtype='datetime64[ns]')})
    correct_frq.set_index('Date', inplace=True)
    correct_frq['hou'] = correct_frq.index.hour
    correct_frq = correct_frq[:data.index[-1]]
    correct_frq = correct_frq[~correct_frq.index.duplicated(keep='first')]
    
    data = pd.concat([correct_frq, data], axis=1)
    data.drop('hou', axis=1, inplace=True)
    data = data[~data.index.duplicated(keep='first')]
    
    return data



def ks_imputation(dataframe, column_to_impute):
    data_smoothed = dataframe.copy()
    #wert_column =  "Wert (kW)" if "Wert (kW)" in data_smoothed.columns else "Wert"
    smoother = KalmanSmoother(component='level_longseason', component_noise={'level':10**-0.5, 'longseason':10**-0.5}, n_longseasons=24*4)
    smoother.smooth(data_smoothed[column_to_impute])
    
    old_val = pd.Series(smoother.data[0], index=data_smoothed.index)
    new_val = pd.Series(smoother.smooth_data[0], index=data_smoothed.index)
    null_index = old_val[old_val.isnull()].index
    
    for item in null_index:
        old_val.loc[item] = new_val.loc[item]
    return data_smoothed
     

def min_max_scale(dataframe, features_to_scale): 
    """Normalization into the scale of [0, 1]
    """
    mm = MinMaxScaler()
    for feature in features_to_scale:
        scaled_featured =  mm.fit_transform(np.array(dataframe[feature]).reshape(-1, 1))
        scaled_featured = [item[0] for item in scaled_featured]
        dataframe[feature] = pd.Series(scaled_featured, index= dataframe.index)
    
    
    
def standardizaton(dataframe, features_to_standardize):
    """
    Standardization into mean 0 and std 1
    """
    for feature in features_to_standardize:
        feature_mean,feature_std = dataframe[feature].mean(),dataframe[feature].std()
        dataframe[feature] = (dataframe[feature] - feature_mean) / feature_std
    
    
    
    
    
    
    
    
    
    
    