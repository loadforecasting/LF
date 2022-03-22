import warnings
warnings.filterwarnings("ignore")


import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import matplotlib.pyplot as plt
from read_data import read_data
from neural_nets.functions import window_dataset
from process import min_max_scale
import argparse
import read_data
import pandas as pd
import add_features

print(tf.__version__)


if __name__=='__main__':
       ## recieve a file
       ## append features
       ## order features is corrert
       ## call the trained model
       pass