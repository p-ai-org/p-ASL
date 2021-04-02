import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional
from sklearn.model_selection import train_test_split
from utilities.reference import *
from utilities.util import *
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=[-1, 1])

''' What type of handsign are we predicting? '''
N_CLASSES = len(MOTIONS)
print(MOTIONS)

TIMESTEPS = 30
FEATURES = 6


# Which corpus to use
CORPUS_DIR = MOTION_CORPUS_DIR

# Whether to save this model or not
SAVE = False
SAVE_FNAME = 'lstm'

def evaluate_cm(classifier, X_test, y_test):
  ''' Makes predictions using classifier and displays confusion matrix of actual v. predicted '''
  y_pred = np.argmax(classifier.predict(X_test), axis=1)
  y_test = np.argmax(y_test, axis=1)
  plot_cm(y_pred, y_test, categories=MOTIONS)

def buildModel(dropout_rate=0.3):
  ''' Build neural network '''
  model = Sequential()
  model.add(LSTM(50, input_shape=(TIMESTEPS, FEATURES)))
  # model.add(Dense(64, activation='relu'))
  # model.add(Dropout(dropout_rate))
  model.add(Dense(32))
  model.add(Dense(N_CLASSES, activation='softmax'))
  return model

def compileModel(model):
  model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

def trainModel(model, X_train, y_train, batch_size=10, epochs=25):
  model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs)

def evaluate(model, X_test, y_test, batch_size=10):
  return model.evaluate(X_test, y_test, batch_size=batch_size)

# Load X and y data
X, y = retrieve_Xy_data(corpus_dir=CORPUS_DIR)

# Convert each y value to a one-hot vector. E.g. 2 (C) --> [0 0 1 0 0 ... 0 0 0]
y = get_one_hot(np.array(y), N_CLASSES)

print(np.max(X), np.min(X))

# Scale y a little bit... janky but
X[:,:,1] *= 1.5
X[:,:,4] *= 1.5

# Check shapes
print(X.shape)
print(y.shape)


# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
# X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

# X_train = X
# y_train = y
# X_test, y_test = retrieve_Xy_data(corpus_dir='data/classifier_data_normalized/corpus/')
# y_test = get_one_hot(np.array(y_test), N_CLASSES)

model = buildModel()
compileModel(model)
trainModel(model, X_train, y_train, epochs=8, batch_size=5)

# Evaluate and visualize
print(evaluate(model, X_test, y_test, batch_size=5))
evaluate_cm(model, X_test, y_test)

# If saving, save model to model directory
if SAVE:
  create_directory_if_needed(MODEL_DIR)
  model.save(f'{MODEL_DIR}{SAVE_FNAME}')