import tensorflow as tf
from functools import reduce
import numpy as np
import pandas as pd
from process import min_max_scale, standardizaton
import matplotlib.pyplot as plt


def split_data(series, train_freq, test_len=17345):
    """Splits input series into train, val and test.
    """
    #slice the last year of data for testing 1 year has 8760 hours
    test_slice = len(series)-test_len

    test_data = series[test_slice:]
    train_val_data = series[:test_slice]

    #make train and validation from the remaining
    train_size = int(len(train_val_data) * train_freq)
    
    train_data = train_val_data[:train_size]
    val_data = train_val_data[train_size:]
    
    return train_data, val_data, test_data



def window_dataset(data, window_size, n_horizon, batch_size, shuffle_buffer, multi_var=False, expand_dims=False):
    
    """ Create a windowed tensorflow dataset
    """

    #create a window with n steps back plus the size of the prediction length
    window = window_size + n_horizon 
    
    #expand dimensions to 3D to fit with LSTM inputs
    
    if expand_dims:
        ds = tf.expand_dims(data, axis=-1)
        ds = tf.data.Dataset.from_tensor_slices(ds)
    else:
        ds = tf.data.Dataset.from_tensor_slices(data)
    
    #create the window function shifting the data by the prediction length
    ds = ds.window(window, shift=n_horizon, drop_remainder=True)
    
    #flatten the dataset and batch into the window size
    ds = ds.flat_map(lambda x : x.batch(window))
    ds = ds.shuffle(shuffle_buffer)    
    
    if multi_var:
        ds = ds.map(lambda x : (x[:-n_horizon], x[-n_horizon:, :1]))
    
    else:
        ds = ds.map(lambda x : (x[:-n_horizon], x[-n_horizon:]))
        #ds = ds.map(lambda x : (x[:-n_horizon,:1], x[-n_horizon:, :1]))
        
    ds = ds.batch(batch_size).prefetch(1)
    
    return ds

def all_dataset_wrapper(frames, args, features_to_scale, model):

    """
    in case of use of multiple datsets, each batch contain homogeneous data 
    """
    train = []
    val =  [] 
    test = []
    for frame in frames:
        ## split here and scale here
        train_data, val_data, test_data = split_data(frame, train_freq=0.7)
        if(args['type_scale']=='normalization'):
            min_max_scale(train_data, features_to_scale)
            min_max_scale(val_data, features_to_scale)
            min_max_scale(test_data, features_to_scale)
        elif(args['type_scale']=='standardization'):
            standardizaton(train_data, features_to_scale)
            standardizaton(val_data, features_to_scale)
            standardizaton(test_data, features_to_scale)
        
        train.append(train_data)
        val.append(val_data)
        test.append(test_data)

    train = pd.concat(train)
    val = pd.concat(val)
    test = pd.concat(test)
    #window_dataset here

    train_ds = window_dataset(train, args['window_size'], args['n_horizon'], args['batch_size'], args['shuffle_buffer_size'], multi_var=args['multi_var'])
    val_ds = window_dataset(val, args['window_size'], args['n_horizon'], args['batch_size'], args['shuffle_buffer_size'], multi_var=args['multi_var'])
    test_ds = window_dataset(test, args['window_size'], args['n_horizon'], args['batch_size'], args['shuffle_buffer_size'], multi_var=args['multi_var'])

    #return reduce(lambda x,y: x.concatenate(y), all_ds)
    model.compile(loss=args['loss'], optimizer=args['optimizer'], metrics=[args['metric']])
    model_history = model.fit(train_ds, validation_data=val_ds, epochs=args['epochs'], verbose=1)
    return model_history, model, test_ds
    
    

class WindowGenerator():

    def __init__(self, input_width, label_width, shift,df,label_columns=None):
        # Store the raw data.
        self.df = df
    
        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                           enumerate(df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
                f'Total window size: {self.total_window_size}',
                f'Input indices: {self.input_indices}',
                f'Label indices: {self.label_indices}',
                f'Label column name(s): {self.label_columns}'])
    
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)
            
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        
        return inputs, labels

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,)

        ds = ds.map(self.split_window)
        return ds

    @property
    def data(self):
        return self.make_dataset(self.df)

   
    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None: 
          # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.data))
      # And cache it for next time
            self._example = result
        return result


def wrapper(data_frame,model,target_column,features_to_scale, args):
    """
    Processes a dataframe and trains a model on it.

            Parameters:
                    data_frame (pd dataframe)
                    target_column (string)
                    features_to_scale (list of strings)
                    type_scale (string): either standardization or normalization
                    input_width (int) : length of lookback timesteps
                    label_width (int) : length of future horizon
                    shift (int ) : timesteps length between input and label
            Returns:
                   model_history : loss and metric history of model training
                   model (tf model) : tensorflow or keras model
    """
    
    train_data,val_data,test_data= split_data(data_frame, train_freq=0.7, test_len=17345)
    if(features_to_scale):
        if(args['type_scale']=='normalization'):
            min_max_scale(train_data, features_to_scale)
            min_max_scale(val_data, features_to_scale)
            min_max_scale(test_data, features_to_scale)
        elif(args['type_scale']=='standardization'):
            standardizaton(train_data, features_to_scale)
            standardizaton(val_data, features_to_scale)
            standardizaton(test_data, features_to_scale)
            
            
    train_ds=WindowGenerator(input_width=args['window_size'],label_width=args['n_horizon'],shift=args['shift'],df=train_data,label_columns=[target_column])
    val_ds=WindowGenerator(input_width=args['window_size'],label_width=args['n_horizon'],shift=args['shift'],df=val_data,label_columns=[target_column])
    test_ds=WindowGenerator(input_width=args['window_size'],label_width=args['n_horizon'],shift=args['shift'],df=test_data,label_columns=[target_column])
    
    model.compile(loss=args['loss'], optimizer=args['optimizer'], metrics=[args['metric']])
    model_history = model.fit(train_ds.data, validation_data=val_ds.data, epochs=args['epochs'], verbose=1)
    return model_history, model, test_ds.data

'''
def all_ds_wrapper(dataframes, model, args):
    all_ds = all_dataset_wrapper(dataframes, args)
    DATASET_SIZE = len(list(all_ds))

    #train test split
    train_size = int(0.7 * DATASET_SIZE)
    val_size = int(0.15 * DATASET_SIZE)
    test_size = int(0.15 * DATASET_SIZE)

    all_ds = all_ds.shuffle(1000)
    train_ds = all_ds.take(train_size) # Creates a Dataset with at most count elements from this dataset.

    test_ds = all_ds.skip(train_size) #Creates a Dataset that skips count elements from this dataset.
    val_ds = test_ds.skip(val_size) 
    test_ds = test_ds.take(test_size)

    model.compile(loss=args['loss'], optimizer=args['optimizer'], metrics=[args['metric']])
    model_history = model.fit(train_ds, validation_data=val_ds, epochs=args['epochs'], verbose=1)
    return model_history, model, test_ds

'''


def plot_history(history):
    mae=history.history['mae']
    loss=history.history['loss']

    epochs=range(len(loss)) # Get number of epochs
    plt.figure()

    epochs_zoom = epochs[100:]
    mae_zoom = mae[100:]
    loss_zoom = loss[100:]

   #------------------------------------------------
   # Plot Zoomed MAE and Loss
   #------------------------------------------------
    plt.plot(epochs_zoom, mae_zoom, 'r')
    plt.plot(epochs_zoom, loss_zoom, 'b')
    plt.title('MAE and Loss')
    plt.xlabel("Epochs")
    plt.ylabel("MAE")
    plt.legend(["MAE", "Loss"])

    plt.figure()
