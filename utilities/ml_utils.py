import numpy as np
from .reference import *
from sklearn.metrics import confusion_matrix
from .confusion_matrix import make_confusion_matrix
from .util import *
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def evaluate_and_plot_cm(classifier, X_test, y_test, labels, keras=True):
  ''' Makes predictions using classifier and displays confusion matrix of actual v. predicted '''
  if keras:
    y_pred = np.argmax(classifier.predict(X_test), axis=1)
    y_test = np.argmax(y_test, axis=1)
  else:
    y_pred = classifier.predict(X_test)
  plot_cm(y_pred, y_test, categories=labels)

def load_data_train_test_suite(model, X, y, labels, batch_size=32, epochs=8, split_size=0.33, verbose=True):
  if verbose:
    # Check shapes
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
  # Split data into train and test
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size)
  model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs)
  # Evaluate and visualize
  loss, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size)
  if verbose:
    print(f"Loss: {loss}; accuracy: {accuracy}")
  evaluate_and_plot_cm(model, X_test, y_test, labels=labels, keras=True)
  return model