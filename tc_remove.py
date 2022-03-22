from datetime import timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import argparse
import read_data

def trend_removal(data, rolling, column_name):
    """
    Function that takes a time series and removes the trend by using a moving average.
    rolling should be chosen to remove long term trends, but keep the daily and monthly seasonalites
    
    data: time series
    rolling : the moving average window

    returns a trend removed time series
    """
    #wert_column = "Wert (kW)" if "Wert (kW)" in data.columns else "Wert"
    df= data.copy()
    #first: trend smoothing using moving average
    df['mov_avg'] = df[column_name].rolling(window=rolling).mean()
    df.mov_avg.fillna(df[column_name], inplace=True)
    
    #second : trend removal using the trend factor
    tsv = df[column_name][-1] # a reference value, eg, the last value
    df['tfi'] = df['mov_avg'] / tsv
    df['wert_trend_rm'] = df[column_name] - df['tfi']
    return df


def generate_data(series, m):
    data_windows = []
    j=0
    i=0
    while i+j<len(series):
        
        data_windows.append([series[j: j+4*m-1]])
        j=j+4*m-1
        i+=1
    return data_windows

## generate data windows using the date utility packages
def generate_data_2(series, seasonality):
    """
    Function that takes a time series and returns daily or monthly data windows .
    
    series: time series
    seasonality : type of seasonality (daily or monthly)
    """
    data_windows = []
    if seasonality=='daily':
        first_day = series.index[0]
        last_day = series.index[-1]
        while(first_day < last_day):
            sw = series[first_day: first_day + timedelta(days=1)]
            sw = sw[:-1]
            data_windows.append(sw)
            first_day= first_day + timedelta(days=1)
        return data_windows
    elif seasonality=='monthly':
        first_month = series.index[0]
        last_month = series.index[-1]
        while(first_month < last_month):
            sw = series[first_month: first_month + relativedelta(months=1)]
            sw = sw[:-1]
            data_windows.append(sw)
            first_month= first_month + relativedelta(months=1)
        return data_windows
    return


def seasonal_indexes(series, seasonality):
    ##the same procedure is followed for each seasonality considered.
    """
    Function that takes a time series and returns seasonal indexes .
    
    series: time series
    seasonality : type of seasonality (daily or monthly)
    
    returns the mean vector of each seasonal window
    """
    data_windows = generate_data_2(series, seasonality)

    if(data_windows != None):
        data_windows[-1] = np.append(data_windows[-1], [0]*(len(data_windows[0])-len(data_windows[-1])))
        y_bar = np.mean([np.mean(item) for item in data_windows])
        p_mean = np.zeros(np.array(data_windows[0]).shape) 
        for i in range(len(p_mean)):
            p_mean[i] = np.mean([item[i] for item in data_windows if i<len(item)]) / y_bar
        return p_mean
        
    return 


def seasonality_removal(data, seasonality,column_name):
    """
    Function that takes a time series data and removes seasonality.
    
    data: time series dataset
    seasonality : type of seasonality (daily or monthly)
    
    returns 
    """
    #wert_column = "wert_trend_rm"
    df = data.copy()
    series = df[column_name]
    first_index = series.index[0]
    last_index = series.index[-1]
    dw = seasonal_indexes(series, seasonality) ## 96 length for daily and 2976 for monthly
    
    frames = []
    while(first_index < last_index):
        if seasonality == 'daily':
            sw = series[first_index: first_index + relativedelta(days=1)]
        elif seasonality == 'monthly':
            sw = series[first_index: first_index + relativedelta(months=1)]
        sw = sw[:-1]
        dww = dw[:len(sw)]
        seasonal_index = pd.Series(dww, index= sw.index)
        pp = pd.DataFrame({'load':sw, 'seasonal_index':seasonal_index})
        frames.append(pp)
        if seasonality == 'daily':
            first_index = first_index + relativedelta(days=1)
        elif seasonality == 'monthly':
            first_index = first_index + relativedelta(months=1)
    frames = pd.concat(frames)
    frames['wert_seasonality_rm'] = frames['load'] - frames ['seasonal_index'] # the ts model is additive
    frames.drop(['load'], axis=1, inplace=True)
    return pd.concat([df, frames], axis=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments get parsed via --commands')
    parser.add_argument("-i", metavar='input file', required=True,help='an input dataset in .excel or .csv file')
    parser.add_argument("-c", metavar='column name', required=True,help='the name of the column that needs trend or/and seasonality removal')
    parser.add_argument("-t", metavar='type of precessing', required=True,help='t for trend, s for seasonalitym ts for both')
    args = parser.parse_args()
    data = read_data.read_data(args.i, 'Zeitstempel', multiple_sheets=True)
    trend_removal(data, 4*24, args.c)
    ts = seasonality_removal(data, 'daily','wert_trend_rm')
    print(ts.head())
