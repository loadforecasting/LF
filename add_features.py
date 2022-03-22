import os
from xml.dom import minicompat
from dotenv import load_dotenv
import urllib.request, urllib.error
import csv, json, sys
import codecs
import datetime
import holidays
import json 
import pandas as pd
import numpy as np
from read_data import read_data
from meteostat import Point, Daily, Hourly
import argparse
import read_data

load_dotenv()
api_key = os.environ['apikey']
de_holidays = holidays.Germany()
'''
#für Kiel
altitude = 5
latitude=54.323334
longitude=10.139444
'''
def add_weather_features(start_date_time,end_date_time, latitude, longitude):
  latitude='{:.5f}'.format(latitude)
  longitude='{:.5f}'.format(longitude)
  weather_api_endpoint = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/weatherdata/history?"
  elements = "datetime%2CdatetimeEpoch%2Ctemp%2Cdew%2Chumidity%2Cprecip%2Cprecipprob%2Cprecipcover%2Cpreciptype%2Csnow%2Cwindgust%2Cwindspeed%2Cwinddir%2Cpressure%2Ccloudcover%2Csolarradiation"
  query_params = '&contentType=json&aggregateMinutes=15&unitGroup=us&includeAstronomy=true&collectStationContributions=true&elements={}&key={}&startDateTime={}&endDateTime={}&locations={},{}'


  column_names = ['datetimeStr','temp','dew','humidity','precip','snow','wdir','wspd','wgust','lpressure','sonnendauer']
  df= pd.DataFrame(columns = column_names)
  query_params=query_params.format(elements, api_key, start_date_time.isoformat(), end_date_time.isoformat(), latitude, longitude)

  try: 
    url = weather_api_endpoint + query_params
    ResultBytes = urllib.request.urlopen(url)
    data = ResultBytes.read()
    weatherData = json.loads(data.decode('utf-8'))
    locations = weatherData['locations']
    #json.dump(weatherData, open("out.json","w"))
    for locationid in locations:  
      location=locations[locationid]
      for value in location["values"]: #value["stationinfo"]
        #print(value['datetimeStr'], value["temp"],value["dew"],value["humidity"],value["precip"],value["snow"],value['wdir'],value['wspd'], value['wgust'],value['sealevelpressure'], value['solarradiation'])
        d1 = datetime.datetime.strptime(value['sunrise'], "%Y-%m-%dT%H:%M:%S%z")
        d2 = datetime.datetime.strptime(value['sunset'], "%Y-%m-%dT%H:%M:%S%z")
        c = abs((d2 - d1))
        sonnendauer = 0
        minutes = divmod(c.total_seconds(), 60) 
        if minutes[1] > 30:
          sonnendauer = minutes[0]+1
        df.loc[len(df)] = [value['datetimeStr'], value["temp"],value["dew"],value["humidity"],value["precip"],value["snow"],value['wdir'],value['wspd'], value['wgust'],value['sealevelpressure'], sonnendauer]
        print('----')
    df.set_index('datetimeStr', inplace=True)
    df.index = df.index.astype('datetime64[ns]')
    return df
  except Exception as r:
    raise Exception("Error {} reading from {}".format(r, weather_api_endpoint+query_params))
  return


def time_of_day(x):
  hour_dict = {'morning': list(np.arange(6,13)),'afternoon': list(np.arange(13,16)), 'evening': list(np.arange(16,22)),
            'night': [22, 23, 0, 1, 2, 3, 4, 5]}
  if x in hour_dict['morning']:
    return 'morning'
  elif x in hour_dict['afternoon']:
    return 'afternoon'
  elif x in hour_dict['evening']:
    return 'evening'
  else:
    return 'night'

def season_calc(month):
  """months from June to September are denoted as 'summer' and months from October to May as 'winter'. """
  if month in [6,7,8,9]:
    return "summer"
  else:
    return "winter"

def return_holiday(date):
  if date in de_holidays:
    return 1
  return 0

def return_holiday_name(date):
  return de_holidays.get(date)

def add_calendar_features(dataframe):
  dataframe.index = dataframe.index.astype('datetime64[ns]')
  data = dataframe.copy()
  weekdays = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3: 'Thursday', 4: 'Friday', 5:'Saturday', 6:'Sunday'}
  data['year'] = data.index.year
  data['month'] = data.index.month
  data['day'] = data.index.day
  data['hour'] = data.index.hour
  data['weekday'] = data.index.weekday.map(weekdays)
  data['dayofweek'] = data.index.dayofweek
  data['weekofyear'] = data.index.isocalendar().week

  data['time_of_day'] = data['hour'].apply(time_of_day)
  data['season'] = data.month.apply(season_calc)
  data['weekend'] = data.apply(lambda x: 1 if ((x['weekday'] in ['Saturday', 'Sunday'])) else 0, axis = 1)
  data['holiday'] = pd.Series(data.index.date, index=data.index, dtype='datetime64[ns]').apply(return_holiday)

  return data

def add_features(dataframe, latitude, longitude):
  # add calendar features
  dataframe = add_calendar_features(dataframe)
  # add weather features
  try:
    weather_features = add_weather_features(dataframe.index[0], dataframe.index[-1], latitude, longitude)
  except Exception as r:
    print("{}".format(r))
    return dataframe
  return dataframe.join(weather_features)


if __name__=='__main__':
  """Read arguments from a command line."""
  parser = argparse.ArgumentParser(description='Arguments get parsed via --commands')
  parser.add_argument("-i", metavar='input file', required=True,help='an input dataset in .excel or .csv file')
  args = parser.parse_args()
  data = read_data.read_data(args.i, 'Zeitstempel', multiple_sheets=True)


  complete_data = add_features(data, 54.323334, 10.139444)
  print(complete_data.head())