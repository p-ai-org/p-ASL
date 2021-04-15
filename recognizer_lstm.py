import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional
from sklearn.model_selection import train_test_split
from utilities.reference import *
from utilities.util import *
import utilities.ml_utils as ml_utils

''' What type of handsign are we predicting? '''
CATEGORIES = MOTIONS

TIMESTEPS = 30
FEATURES = 6

# Which corpus to use
CORPUS_DIR = MOTION_CORPUS_DIR

# Whether to save this model or not
SAVE = True
SAVE_FNAME = 'motion_lstm'

def buildModel(dropout_rate=0.3):
  ''' Build neural network '''
  model = Sequential()
  model.add(LSTM(100, input_shape=(TIMESTEPS, FEATURES)))
  model.add(Dense(64))
  model.add(Dropout(dropout_rate))
  model.add(Dense(32))
  model.add(Dense(len(CATEGORIES), activation='softmax'))
  return model

def buildBaselineModel(dropout_rate=0.3):
  ''' Build neural network '''
  model = Sequential()
  model.add(LSTM(100, input_shape=(TIMESTEPS, FEATURES)))
  model.add(Dense(64))
  model.add(Dropout(dropout_rate))
  model.add(Dense(32))
  model.add(Dense(N_CLASSES, activation='softmax'))
  return model

# Load X and y data
X, y = retrieve_Xy_data(corpus_dir=CORPUS_DIR)
# Convert each y value to a one-hot vector. E.g. 2 (C) --> [0 0 1 0 0 ... 0 0 0]
y = get_one_hot(np.array(y), len(CATEGORIES))
# Scale y a little bit... janky but
X[:,:,1] *= 1.5
X[:,:,4] *= 1.5

EPOCHS = 8
BATCH_SIZE = 5
model = buildModel()
model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model = ml_utils.load_data_train_test_suite(model, X, y, labels=CATEGORIES, batch_size=BATCH_SIZE, epochs=EPOCHS)

# If saving, save model to model directory
if SAVE:
  create_directory_if_needed(MODEL_DIR)
  model.save(f'{MODEL_DIR}{SAVE_FNAME}')