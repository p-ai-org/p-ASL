import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from utilities.reference import *
from utilities.util import *

''' What type of handsign are we predicting? '''
# CATEGORIES = LETTERS
CATEGORIES = CLASSIFIERS
N_CLASSES = len(CATEGORIES)

# Which corpus to use
# CORPUS_DIR = LETTER_CORPUS_DIR
# CORPUS_DIR = CLASSIFIER_CORPUS_DIR
CORPUS_DIR = CLASSIFIER_NORM_CORPUS_DIR

# Whether to save this model or not
SAVE = True
SAVE_FNAME = 'classifier_norm'

def evaluate_cm(classifier, X_test, y_test):
  ''' Makes predictions using classifier and displays confusion matrix of actual v. predicted '''
  y_pred = np.argmax(classifier.predict(X_test), axis=1)
  y_test = np.argmax(y_test, axis=1)
  plot_cm(y_pred, y_test, categories=CATEGORIES)

def buildModel(dropout_rate=0.3):
  ''' Build neural network '''
  model = Sequential()
  model.add(Dense(NUM_DIM * NUM_POINTS))
  model.add(Dense(256))
  model.add(Dropout(dropout_rate))
  model.add(Dense(128))
  model.add(Dense(32))
  model.add(Dropout(dropout_rate))
  model.add(Dense(N_CLASSES, activation='softmax'))
  return model

def compileModel(model):
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

def trainModel(model, X_train, y_train, batch_size=32, epochs=25):
  model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs)

def evaluate(model, X_test, y_test, batch_size=32):
  return model.evaluate(X_test, y_test, batch_size=batch_size)

# Load X and y data
X, y = retrieve_Xy_data(corpus_dir=CORPUS_DIR)

# Convert each y value to a one-hot vector. E.g. 2 (C) --> [0 0 1 0 0 ... 0 0 0]
y = get_one_hot(np.array(y), N_CLASSES)

# Check shapes
print(X.shape)
print(y.shape)

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# X_train = X
# y_train = y
# X_test, y_test = retrieve_Xy_data(corpus_dir='data/classifier_data_normalized/corpus/')
# y_test = get_one_hot(np.array(y_test), N_CLASSES)

model = buildModel()
compileModel(model)
trainModel(model, X_train, y_train, epochs=8)

# Evaluate and visualize
print(evaluate(model, X_test, y_test))
evaluate_cm(model, X_test, y_test)

# If saving, save model to model directory
if SAVE:
  create_directory_if_needed(MODEL_DIR)
  model.save(f'{MODEL_DIR}{SAVE_FNAME}')