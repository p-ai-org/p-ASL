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

def encoder_decoder_predict(infenc, infdec, source, n_steps, cardinality):
  ''' Make a prediction on X input data with an encoder and decoder '''
  # encode
  state = infenc.predict(source)
  # start of sequence input
  target_seq = np.array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
  # collect predictions
  output = list()
  for t in range(n_steps):
    # predict next char
    yhat, h, c = infdec.predict([target_seq] + state)
    # store prediction
    output.append(yhat[0,0,:])
    # update state
    state = [h, c]
    # update target sequence
    target_seq = yhat
  return np.array(output)

def token_lookup(i, location):
  with open(location, 'r') as f:
    tokens = f.read().split('\n')[:-1]
  return tokens[i]

def multiple_token_lookup(indexes, location):
  res = []
  with open(location, 'r') as f:
    tokens = f.read().split('\n')[:-1]
  for i in indexes:
    res.append(tokens[i])
  return res

def plot_loss(history, n_epochs):
  loss_train = history.history['loss']
  loss_val = history.history['val_loss']
  epochs = range(1, n_epochs + 1)
  plt.plot(epochs, loss_train, 'g', label='Training loss')
  plt.plot(epochs, loss_val, 'b', label='Validation loss')
  plt.title('Training and Validation loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.show()

def plot_accuracy(history, n_epochs):
  accuracy_train = history.history['accuracy']
  accuracy_val = history.history['val_accuracy']
  epochs = range(1, n_epochs + 1)
  plt.plot(epochs, accuracy_train, 'g', label='Training accuracy')
  plt.plot(epochs, accuracy_val, 'b', label='Validation accuracy')
  plt.title('Training and Validation accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.show()