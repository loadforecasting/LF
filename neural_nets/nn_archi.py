import tensorflow as tf

def fnn_model(window_size,n_features,n_horizon):
    """
    A normal fully connected 3 layers neural network
    """
    inputs = tf.keras.layers.Input(shape=(window_size,n_features), name='main')
    layer_1 = tf.keras.layers.Dense(20, activation="relu")(inputs)
    layer_2 = tf.keras.layers.Dense(10, activation="relu")(layer_1)
    outputs = tf.keras.layers.Dense(n_horizon)(layer_2)
    ff = tf.keras.Model(inputs=inputs, outputs=outputs, name='vanilla_nn')
    return ff

def bi_lstm(window_size,n_features,n_horizon):
    inputs = tf.keras.layers.Input(shape=(window_size, n_features), name='main')
    layer_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True))(inputs)
    layer_2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(layer_1)
    dense_1 = tf.keras.layers.Dense(n_horizon)(layer_2)
    outputs = tf.keras.layers.Lambda(lambda x: x * 100.0)(dense_1)
    bi_lstm = tf.keras.Model(inputs=inputs, outputs=outputs, name='bi_lstm')
    return bi_lstm

def cnn_model(window_size,n_features,n_horizon):
    inputs = tf.keras.layers.Input(shape=(window_size, n_features), name='main')
    layer_1 =tf.keras.layers.Conv1D(64, kernel_size=6, activation='relu')(inputs)
    pool_1 = tf.keras.layers.MaxPooling1D(2)(layer_1)

    layer_2 = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu')(pool_1)
    pool_2 = tf.keras.layers.MaxPooling1D(2)(layer_2)

    flatten = tf.keras.layers.Flatten()(pool_2)
    drop_1 = tf.keras.layers.Dropout(0.3)(flatten)
    dense_1 = tf.keras.layers.Dense(128)(drop_1)
    drop_2 = tf.keras.layers.Dropout(0.3)(dense_1)
    outputs = tf.keras.layers.Dense(n_horizon)(drop_2)
    cnn_mod = tf.keras.Model(inputs=inputs, outputs=outputs, name='vanilla_cnn')
    return cnn_mod

class ResidualWrapper(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def call(self, inputs, *args, **kwargs):
        delta = self.model(inputs, *args, **kwargs)

    # The prediction for each time step is the input
    # from the previous time step plus the delta
    # calculated by the model.
        return inputs + delta
    
residual_lstm = ResidualWrapper(
    tf.keras.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.Dense(1,kernel_initializer=tf.initializers.zeros())]))

def conv_lstm_model(window_size,n_features,n_horizon):
    CONV_WIDTH=3
    inputs = tf.keras.layers.Input(shape=(window_size, n_features), name='main')
    layer_1 =tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :])(inputs)
    cnv_layer = tf.keras.layers.Conv1D(96, activation='relu', kernel_size=(CONV_WIDTH))(layer_1)
    lstm_1 = tf.keras.layers.LSTM(60, return_sequences=True)(cnv_layer)
    drop_1 = tf.keras.layers.Dropout(0.3)(lstm_1)
    lstm_2 = tf.keras.layers.LSTM(60, return_sequences=True)(drop_1)
    dense_1 = tf.keras.layers.Dense(128, activation='relu')(lstm_2)
    outputs = tf.keras.layers.Dense(n_horizon)(dense_1)
    conv_lstm_model = tf.keras.Model(inputs=inputs, outputs=outputs, name='conv_lstm_model')
    return conv_lstm_model

def cnn_lstm_skip(window_size,n_features,n_horizon):

    inputs = tf.keras.layers.Input(shape=(window_size, n_features), name='main')
        
    conv1 = tf.keras.layers.Conv1D(64, kernel_size=6, activation='relu')(inputs)
    max_pool_1 = tf.keras.layers.MaxPooling1D(2)(conv1)

    conv2 = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu')(max_pool_1)
    max_pool_2 = tf.keras.layers.MaxPooling1D(2)(conv2)
        
    lstm_1 = tf.keras.layers.LSTM(72, activation='relu', return_sequences=True)(max_pool_2)
    lstm_2 = tf.keras.layers.LSTM(48, activation='relu', return_sequences=False)(lstm_1)
    
    flatten = tf.keras.layers.Flatten()(lstm_2)
        
    skip_flatten = tf.keras.layers.Flatten()(inputs)

    concat = tf.keras.layers.Concatenate(axis=-1)([flatten, skip_flatten])
    drop_1 = tf.keras.layers.Dropout(0.3)(concat)
    dense_1 = tf.keras.layers.Dense(128, activation='relu')(drop_1)
    drop_2 = tf.keras.layers.Dropout(0.3)(dense_1)
    output = tf.keras.layers.Dense(n_horizon)(drop_2)
        
    cnn_lstm_skip = tf.keras.Model(inputs=inputs, outputs=output, name='cnn_lstm_skip')

    return cnn_lstm_skip


def vanilla_lstm(window_size,n_features,n_horizon):

    inputs = tf.keras.layers.Input(shape=(window_size, n_features), name='main')
    layer_1 =tf.keras.layers.LSTM(72, activation='relu', return_sequences=True)(inputs)
    #layer_2 = tf.keras.layers.LSTM(48, activation='relu', return_sequences=False)(layer_1)
    flatten = tf.keras.layers.Flatten()(layer_1)
    #drop_1 = tf.keras.layers.Dropout(0.3)(flatten)
    dense_1 = tf.keras.layers.Dense(128, activation='relu')(flatten)
    drop_2 = tf.keras.layers.Dropout(0.3)(dense_1)
    outputs = tf.keras.layers.Dense(n_horizon)(drop_2)

    vani_lstm = tf.keras.Model(inputs=inputs, outputs=outputs, name='vanilla_lstm')

    return vani_lstm

def cnn_bn(window_size, n_features, n_horizon, pre_model=None):
    inputs = tf.keras.layers.Input(shape=(window_size, n_features), name='main')
    conv_1 =tf.keras.layers.Conv1D(128, kernel_size=8, padding='same')(inputs)
    conv_1 = tf.keras.layers.BatchNormalization()(conv_1)
    conv_1 = tf.keras.layers.Activation(activation='relu')(conv_1)

    conv_2 =tf.keras.layers.Conv1D(256, kernel_size=5, padding='same')(conv_1)
    conv_2 = tf.keras.layers.BatchNormalization()(conv_2)
    conv_2 = tf.keras.layers.Activation(activation='relu')(conv_2)

    conv_3 =tf.keras.layers.Conv1D(128, kernel_size=5, padding='same')(conv_2)
    conv_3 = tf.keras.layers.BatchNormalization()(conv_3)
    conv_3 = tf.keras.layers.Activation(activation='relu')(conv_3)

    gap_layer = tf.keras.layers.GlobalAveragePooling1D()(conv_3)

    outputs = tf.keras.layers.Dense(n_horizon)(gap_layer)
    cnn_bn = tf.keras.Model(inputs=inputs, outputs=outputs, name='cnn_bn')

    if pre_model is not None:
        for i in range(len(cnn_bn.layers)-1):
            cnn_bn.layers[i].set_weights(pre_model.layers[i].get_weights())

    return cnn_bn