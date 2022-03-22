from json import load
import os
from add_features import add_features
from read_data import read_data
from process import correct_frequency, ks_imputation
import pandas as pd
from neural_nets.functions import all_dataset_wrapper, plot_history, wrapper
from neural_nets import nn_archi
import args
import tensorflow as tf
from os import walk
import matplotlib.pyplot as plt
import variables
import add_features

file_path = variables.one_file_path
file_dir = variables.file_dir
all_files = next(walk(file_dir), (None, None, []))[2]  # [] if no file


features_to_drop = variables.features_to_drop
features_to_scale = variables.features_to_scale
features_to_dummify = variables.features_to_dummify
   
argss = {'epochs': 1,'loss':tf.keras.losses.Huber(),'optimizer':tf.keras.optimizers.Adam(learning_rate=1e-6),'metric':'mae',
               'window_size':96, 'n_horizon':1, 'batch_size':128, 'shuffle_buffer_size': 1000, 'multi_var':True, 'type_scale':'normalization','shift':1, 'target_column':'Wert (kW)'
}

one_file = True

if __name__ == "__main__":
   
   if(one_file == True):
      data = read_data(file_path, 'Zeitstempel',multiple_sheets=True)
      #data = correct_frequency(data)
      #data = ks_imputation(data)
   
      data.drop(features_to_drop, axis = 1, inplace=True, errors='ignore')
      data = pd.get_dummies(data, columns = features_to_dummify, drop_first=True)
      if('Unnamed: 0') in data.columns:
         data.drop(['Unnamed: 0'], axis=1, inplace=True)
      load_column = "Wert (kW)" if "Wert (kW)" in data.columns else "Wert"
      n_features = data.shape[1]
      nn = nn_archi.conv_lstm_model(window_size=argss['window_size'], n_features=n_features, n_horizon=argss['n_horizon'])
      model_history, model, test_ds = wrapper(data, nn,load_column,features_to_scale, argss)
   
   else:
      frames = [] 
      for file in all_files:
         df = read_data(file_dir+'/'+file,'Zeitstempel',multiple_sheets=True)
         df.drop(features_to_drop, axis = 1, inplace=True,  errors='ignore')
         df = pd.get_dummies(df, columns = features_to_dummify, drop_first=True)
         if('Unnamed: 0') in df.columns:
            df.drop(['Unnamed: 0'], axis=1, inplace=True)
         
         frames.append(df)
      print('n_features : ',frames[0].shape[1])
      print('window_size :', argss['window_size'])
      print(frames[0].head())
      n_features = frames[0].shape[1]

      nn = nn_archi.conv_lstm_model(window_size=argss['window_size'], n_features=n_features, n_horizon=argss['n_horizon'])
      model_history, model, test_ds = all_dataset_wrapper(frames, argss, features_to_scale, nn)

   plot_history(model_history)
   #================================= Test Data ===========================================#
   print('Evaluation on test data')
   performance = model.evaluate(test_ds)
   print('performance: ', performance)
   #================================= save model weights ==================================#
   model_weights_dir = '../model_weights/1_horizons/one_day_lookback/'+model.name
   os.makedirs(model_weights_dir, exist_ok=False)
   model.save(model_weights_dir)

