import numpy as np
from .letters import CLASS_TO_LETTER

FPS = 30
NUM_DIM = 3
NUM_POINTS = 21

LETTER_DATA_DIR = 'letter_data/'
CORPUS_DIR = LETTER_DATA_DIR + 'corpus/'
MODEL_DIR = 'models/'

def unit_vector(vector):
  """ Returns the unit vector of the vector.  """
  return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
  """ Returns the angle in radians between vectors 'v1' and 'v2'::
  >>> angle_between((1, 0, 0), (0, 1, 0))
  1.5707963267948966
  >>> angle_between((1, 0, 0), (1, 0, 0))
  0.0
  >>> angle_between((1, 0, 0), (-1, 0, 0))
  3.141592653589793
  """
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

def create_Xy_data(save=False):
  X = np.zeros((1, NUM_POINTS * NUM_DIM))
  y = []
  for i, letter in enumerate(list(CLASS_TO_LETTER.values())):
    data = np.load('{}{}.npy'.format(LETTER_DATA_DIR, letter))
    X = np.concatenate((X, flatten_np_hand(data)))
    y += [i] * len(data)
  y = np.array(y)
  return X[1:], y

def save_Xy_data():
  X, y = create_Xy_data()
  np.save('{}X.npy'.format(CORPUS_DIR), X)
  np.save('{}y.npy'.format(CORPUS_DIR), y)

def retrieve_Xy_data():
  return np.load('{}X.npy'.format(CORPUS_DIR)), np.load('{}y.npy'.format(CORPUS_DIR))
