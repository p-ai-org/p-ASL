import numpy as np
from .reference import *
from sklearn.metrics import confusion_matrix
from .confusion_matrix import make_confusion_matrix
import matplotlib.pyplot as plt
import os

NUM_DIM = 3
NUM_POINTS = 21

DATA_DIR = 'data/'

LETTER_DATA_DIR = DATA_DIR + 'letter_data/'
LETTER_CORPUS_DIR = LETTER_DATA_DIR + 'corpus/'

CLASSIFIER_DATA_DIR = DATA_DIR + 'classifier_data/'
CLASSIFIER_CORPUS_DIR = CLASSIFIER_DATA_DIR + 'corpus/'

CLASSIFIER_NORM_DATA_DIR = DATA_DIR + 'classifier_norm_data/'
CLASSIFIER_NORM_CORPUS_DIR = CLASSIFIER_NORM_DATA_DIR + 'corpus/'
MODEL_DIR = 'models/'

def unit_vector(vector):
  """ Returns the unit vector of the vector.  """
  return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
  """ Returns the angle in radians between vectors 'v1' and 'v2' """
  v1_u = unit_vector(v1)
  v2_u = unit_vector(v2)
  return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def normalize_np_hand(arr):
  mean = get_center_of_np_hand(arr)
  centered = np.subtract(arr, mean)
  max_mag = max([np.linalg.norm(x) for x in centered[0]])
  norm = np.divide(centered, max_mag)
  return norm

def get_center_of_np_hand(arr):
  return np.mean(arr, axis=1)[0]

def get_x_rot_of_np_hand(arr):
  knuckle = arr[9]
  wrist = arr[0]
  knuckle_point = -np.array([knuckle.x - wrist.x, knuckle.y - wrist.y, knuckle.z - wrist.z])
  angle = angle_between(knuckle_point, np.array([0, 1, 0]))
  return angle if knuckle.z < wrist.z else -angle

def landmarks_to_np(landmark, apply_normalization=True):
  arr = np.empty((1, NUM_POINTS, NUM_DIM))
  for i, point in enumerate(landmark):
    arr[0,i,0] = point.x
    arr[0,i,1] = point.y
    arr[0,i,2] = point.z
  if apply_normalization:
    return normalize_np_hand(arr)
  else:
    return arr

def flatten_np_hand(arr):
  return np.reshape(arr, (arr.shape[0], NUM_POINTS * NUM_DIM))

def landmarks_to_np_flat(landmark, apply_normalization=True):
  arr = landmarks_to_np(landmark, apply_normalization=apply_normalization)
  return flatten_np_hand(arr)

def z_color_fn(x, y, z):
  zed = max(0, min((z * -5 + .5) * 255, 255))
  return (zed, zed, zed)

def euclidian_distance(v1, v2, exaggerate_z=False):
  if v1 is None or v2 is None:
    return None
  if exaggerate_z:
    vec1 = np.array([v1[0], v1[1], v1[2] * 2])
    vec2 = np.array([v2[0], v2[1], v2[2] * 2])
    return np.linalg.norm(vec2 - vec1)
  else:
    return np.linalg.norm(v2 - v1)

def create_Xy_data(save=False, names=LETTERS, data_dir=LETTER_DATA_DIR):
  X = np.zeros((1, NUM_POINTS * NUM_DIM))
  y = []
  for i, name in enumerate(names):
    data = np.load(f'{data_dir}{name}.npy')
    X = np.concatenate((X, flatten_np_hand(data)))
    y += [i] * len(data)
  y = np.array(y)
  return X[1:], y

def save_Xy_data(corpus_dir=LETTER_CORPUS_DIR, names=LETTERS, data_dir=LETTER_DATA_DIR):
  X, y = create_Xy_data(names=names, data_dir=data_dir)
  create_directory_if_needed(corpus_dir)
  np.save(f'{corpus_dir}X.npy', X)
  np.save(f'{corpus_dir}y.npy', y)

def retrieve_Xy_data(corpus_dir=LETTER_CORPUS_DIR):
  return np.load(f"{corpus_dir}X.npy"), np.load(f"{corpus_dir}y.npy")

def plot_cm(y_pred, y_test, categories=LETTERS):
  cm = confusion_matrix(y_test, y_pred)
  make_confusion_matrix(cm, categories=categories, percent=False)
  plt.show()

def get_one_hot(targets, n_classes):
    res = np.eye(n_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[n_classes])

def normalize_hand_angle(hand):
  ''' Normalizes the hand angle such that the vector between points 0 and 1 are aligned with the vertical 
        Parameters:
          hand: a (21, 3) matrix containing the landmarks of a hand
        Returns a tuple (new_hand, angle)
          new_hand: a (21, 3) angle-normalized version of the original hand
          angle: a 3-tuple with the amount of original rotation in the x-, y-, and z- dimensions'''
  new_hand = hand.copy()
  angle = (0, 0, 0)
  return new_hand, angle

def create_directory_if_needed(dirname, verbose=False):
  if os.path.isdir(dirname):
    if verbose:
      print(f"[create_directory_if_needed] Directory '{dirname}' already exists. Aborting.")
    return
  os.makedirs(dirname)
  if verbose:
    print(f"[create_directory_if_needed] Directory '{dirname}' created.")