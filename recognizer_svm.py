from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from utilities.letters import CLASS_TO_LETTER
from sklearn.metrics import confusion_matrix
from utilities.confusion_matrix import make_confusion_matrix
from joblib import dump
from utilities.util import *

N_CLASSES = len(CLASS_TO_LETTER)

def plot_cm(classifier, X_test, y_test):
  class_names = list(CLASS_TO_LETTER.values())
  y_pred = classifier.predict(X_test)
  cm = confusion_matrix(y_test, y_pred)
  make_confusion_matrix(cm, categories=class_names, percent=False)
  plt.show()

def trainSVM(X_train, y_train):
  model = SVC(gamma='auto', probability=True)
  print("Training model... (this can take a while)")
  model.fit(X_train, y_train)
  return model

def evaluate(model, X_test, y_test, batch_size=32):
  return model.evaluate(X_test, y_test, batch_size=batch_size)

X, y = retrieve_Xy_data()

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

model = trainSVM(X_train, y_train)
plot_cm(model, X_test, y_test)

dump(model, '{}recognizer_svm'.format(MODEL_DIR))