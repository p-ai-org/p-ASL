from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from joblib import dump
from utilities.reference import *
from utilities.util import *

''' What type of handsign are we predicting? '''
CATEGORIES = LETTERS
# CATEGORIES = CLASSIFIERS
N_CLASSES = len(CATEGORIES)

''' Which corpus are we using? '''
CORPUS_DIR = LETTER_CORPUS_DIR
# CORPUS_DIR = CLASSIFIER_CORPUS_DIR
# CORPUS_DIR = CLASSIFIER_NORM_CORPUS_DIR

# Whether to save this model or not
SAVE = False
SAVE_FNAME = 'recognizer_svm'

def evaluate_cm(classifier, X_test, y_test):
  y_pred = classifier.predict(X_test)
  plot_cm(y_pred, y_test, categories=CATEGORIES)

def trainSVM(X_train, y_train):
  model = SVC(gamma='auto', probability=True)
  print("Training model... (this can take a while)")
  model.fit(X_train, y_train)
  return model

def evaluate(model, X_test, y_test, batch_size=32):
  return model.evaluate(X_test, y_test, batch_size=batch_size)

X, y = retrieve_Xy_data(corpus_dir=CORPUS_DIR)

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

model = trainSVM(X_train, y_train)
evaluate_cm(model, X_test, y_test)

if SAVE:
  create_directory_if_needed(MODEL_DIR)
  dump(model, f'{MODEL_DIR}{SAVE_FNAME}')