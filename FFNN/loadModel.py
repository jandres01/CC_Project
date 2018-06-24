#Script to load Model

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense, Dropout
from keras.utils import np_utils
import pandas as pd
import numpy as np
import argparse
import cv2
import os

model = Sequential()
model = model.load("FFNN_whole_model.h5")


