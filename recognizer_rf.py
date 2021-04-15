from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from utilities.reference import *
from utilities.util import *
import utilities.ml_utils as ml_utils

''' What type of handsign are we predicting? '''
# CATEGORIES = LETTERS
CATEGORIES = CLASSIFIERS

''' Which corpus are we using? '''
# CORPUS_DIR = LETTER_CORPUS_DIR
CORPUS_DIR = CLASSIFIER_CORPUS_DIR
# CORPUS_DIR = CLASSIFIER_NORM_CORPUS_DIR

# Whether to save this model or not
SAVE = True
SAVE_FNAME = 'svm_non_norm'

def trainRF(X_train, y_train):
  model = RandomForestClassifier()
  print("Training model... (this can take a while)")
  model.fit(X_train, y_train)
  return model

X, y = retrieve_Xy_data(corpus_dir=CORPUS_DIR)
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
model = trainRF(X_train, y_train)
ml_utils.evaluate_and_plot_cm(model, X_test, y_test, labels=CATEGORIES, keras=False)

if SAVE:
  create_directory_if_needed(MODEL_DIR)
  dump(model, f'{MODEL_DIR}{SAVE_FNAME}')