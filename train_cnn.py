import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from utilities.reference import *
from utilities.util import *
import utilities.ml_utils as ml_utils

''' What type of handsign are we predicting? '''
# CATEGORIES = LETTERS
CATEGORIES = CLASSIFIERS

''' Which corpus are we using? '''
# DATASET_DIR = LETTER_DIR
DATASET_DIR = CLASSIFIER_ANYANGLE_DIR
# DATASET_DIR = CLASSIFIER_FORCED_DIR
# DATASET_DIR = CLASSIFIER_UPRIGHT_DIR

CORPUS_DIR = DATASET_DIR + CORPUS_SUFFIX

# Whether to save this model or not
SAVE = False
SAVE_FNAME = 'cnn_model'

def buildModel(dropout_rate=0.3):
  ''' Build neural network '''
  model = Sequential()
  model.add(Dense(NUM_DIM * NUM_POINTS))
  model.add(Dense(256))
  model.add(Dropout(dropout_rate))
  model.add(Dense(128))
  model.add(Dense(32))
  model.add(Dropout(dropout_rate))
  model.add(Dense(len(CATEGORIES), activation='softmax'))
  return model

# Load X and y data
X, y = retrieve_Xy_data(corpus_dir=CORPUS_DIR)
# Convert each y value to a one-hot vector. E.g. 2 (C) --> [0 0 1 0 0 ... 0 0 0]
y = get_one_hot(np.array(y), len(CATEGORIES))

BATCH_SIZE = 32
EPOCHS = 8
model = buildModel()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model = ml_utils.load_data_train_test_suite(model, X, y, labels=CATEGORIES, batch_size=BATCH_SIZE, epochs=EPOCHS)

# If saving, save model to model directory
if SAVE:
  create_directory_if_needed(MODEL_DIR)
  model.save(f'{MODEL_DIR}{SAVE_FNAME}')