import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utilities.letters import CLASS_TO_LETTER
from sklearn.metrics import confusion_matrix
from utilities.confusion_matrix import make_confusion_matrix
from utilities.util import *

N_CLASSES = len(CLASS_TO_LETTER)

def plot_cm(classifier, X_test, y_test):
  class_names = list(CLASS_TO_LETTER.values())
  y_pred = np.argmax(classifier.predict(X_test), axis=1)
  y_test = np.argmax(y_test, axis=1)
  cm = confusion_matrix(y_test, y_pred)
  make_confusion_matrix(cm, categories=class_names, percent=False)
  plt.show()

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def buildModel(dropout_rate=0.3):
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

X, y = retrieve_Xy_data()

y = get_one_hot(np.array(y), N_CLASSES)

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
model = buildModel()
compileModel(model)
trainModel(model, X_train, y_train, epochs=5)
print(evaluate(model, X_test, y_test))
plot_cm(model, X_test, y_test)

model.save('{}recognizer_nn'.format(MODEL_DIR))