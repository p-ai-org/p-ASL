import numpy as np
from .reference import *
from sklearn.metrics import confusion_matrix
from .confusion_matrix import make_confusion_matrix
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

NUM_DIM = 3
NUM_POINTS = 21

DATA_DIR = 'data/'

LETTER_DATA_DIR = DATA_DIR + 'letter_data/'
LETTER_CORPUS_DIR = LETTER_DATA_DIR + 'corpus/'

CLASSIFIER_DATA_DIR = DATA_DIR + 'classifier_data/'
CLASSIFIER_CORPUS_DIR = CLASSIFIER_DATA_DIR + 'corpus/'

CLASSIFIER_NORM_DATA_DIR = DATA_DIR + 'classifier_norm_data/'
CLASSIFIER_NORM_CORPUS_DIR = CLASSIFIER_NORM_DATA_DIR + 'corpus/'

MOTION_DATA_DIR = DATA_DIR + 'motion_data/'
MOTION_CORPUS_DIR = MOTION_DATA_DIR + 'corpus/'

MODEL_DIR = 'models/'

# Define canonical directions in the order that MediaPipe presents it
UP = np.array([0, 1, 0])
RIGHT = np.array([1, 0, 0])
OUT = np.array([0, 0, 1])

def unit_vector(vector):
  ''' Returns the unit vector of the original vector '''
  return vector / np.linalg.norm(vector)

def project_vector_onto_plane(u, n):
  ''' Returns a vector that is the projection of vector u onto the plane defined by normal vector n '''
  n_norm = np.linalg.norm(n)
  u_on_n = (np.dot(u, n) / (n_norm ** 2)) * n 
  projection = u - u_on_n
  return projection

def angle_between(v1, v2):
  ''' Returns the angle in radians between vectors v1 and v2 '''
  v1_u = unit_vector(v1)
  v2_u = unit_vector(v2)
  return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def angle_between_360(v1, v2, n):
  ''' Returns the angle in radians between vectors v1 and v2 along the plane given by normal vector n  
      The result will be in the range [0, 2*pi) '''
  cross = np.cross(v1, v2)
  dot = np.dot(v1, v2)
  angle = np.arctan2(np.linalg.norm(cross), dot)
  test = np.dot(n, cross)
  if test < 0: 
    angle = -angle
  return angle

def rotate_around_right(vector, theta):
  ''' Rotate a vector around the "right" axis '''
  R = np.array([RIGHT, [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
  return np.dot(R,vector)

def rotate_around_up(vector, theta):
  ''' Rotate a vector around the "up" axis '''
  R = np.array([[np.cos(theta), 0, np.sin(theta)], UP, [-np.sin(theta), 0, np.cos(theta)]])
  return np.dot(R,vector)

def rotate_around_out(vector, theta):
  ''' Rotate a vector around the "out" axis '''
  R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], OUT])
  return np.dot(R,vector)

def get_normal_from_three_pts(p1, p2, p3):
  ''' Given three points (assumed counterclockwise order), return the normal vector of the plane formed '''
  direction = np.cross(p2 - p1, p3 - p1)
  norm = unit_vector(direction)
  return norm

def normalize_size(hand):
  ''' Translates the hand so its center lies at (0, 0, 0)  
      Also scales such that the largest vector has magnitude 1 '''
  mean = get_center_of_hand(hand)
  centered = np.subtract(hand, mean)
  max_mag = max([np.linalg.norm(x) for x in centered])
  norm = np.divide(centered, max_mag)
  return norm

def get_center_of_hand(arr):
  ''' Returns the mean (center) of a hand as an array of 3 coordinates '''
  return np.mean(arr, axis=0)

def get_x_rot(arr):
  ''' Finds the rotation (in radians) of the hand forward-backward '''
  knuckle = arr[9]
  wrist = arr[0]
  knuckle_point = -np.array([knuckle.x - wrist.x, knuckle.y - wrist.y, knuckle.z - wrist.z])
  angle = angle_between(knuckle_point, UP)
  return angle if knuckle.z < wrist.z else -angle

def landmarks_to_np(landmark):
  ''' Converts a landmark object from MediaPipe into a numpy array. Returns array with shape (21, 3) '''
  arr = np.empty((NUM_POINTS, NUM_DIM))
  for i, point in enumerate(landmark):
    arr[i,0] = point.x
    arr[i,1] = point.y
    arr[i,2] = point.z
  return arr

def pose_landmarks_to_np(landmark, top_half=True):
  arr = np.empty((len(landmark), NUM_DIM))
  for i, point in enumerate(landmark):
    arr[i,0] = point.x
    arr[i,1] = point.y
    arr[i,2] = point.z
  if top_half:
    arr = arr[:25]
  return arr

def flatten_hand(arr):
  ''' Flattens hands in (21, 3) to (63) '''
  return np.reshape(arr, (NUM_POINTS * NUM_DIM,))

def euclidian_distance(v1, v2, z_scale=1):
  ''' Finds the euclidian distance between the "tips" of v1 and v2
      Includes an option to scale the z (MediaPipe seems to slightly underestimate depth) '''
  if v1 is None or v2 is None:
    return None
  if z_scale != 1:
    vec1 = np.array([v1[0], v1[1], v1[2] * z_scale])
    vec2 = np.array([v2[0], v2[1], v2[2] * z_scale])
    return np.linalg.norm(vec2 - vec1)
  else:
    return np.linalg.norm(v2 - v1)

def create_Xy_data(names=LETTERS, data_dir=LETTER_DATA_DIR):
  ''' Compile all individual sign numpy files into X and y data '''
  X = np.zeros((1, NUM_POINTS * NUM_DIM))
  y = []
  for i, name in enumerate(names):
    data = np.load(f'{data_dir}{name}.npy')
    X = np.concatenate((X, np.reshape(data, (data.shape[0], NUM_POINTS * NUM_DIM))))
    y += [i] * len(data)
  y = np.array(y)
  return X[1:], y

def create_Xy_data_motion(timesteps=30):
  ''' Compile all individual motion numpy files into X and y data '''
  X = np.zeros((1, timesteps, NUM_DIM * 2))
  y = []
  for i, name in enumerate(MOTIONS):
    data = np.load(f'{MOTION_DATA_DIR}{name}.npy')
    X = np.concatenate((X, data))
    y += [i] * len(data)
  y = np.array(y)
  return X[1:], y

def save_Xy_data(corpus_dir=LETTER_CORPUS_DIR, names=LETTERS, data_dir=LETTER_DATA_DIR):
  ''' Compiles all the numpy files in data_dir into X.npy and y.npy and saves to corpus_dir '''
  X, y = create_Xy_data(names=names, data_dir=data_dir)
  create_directory_if_needed(corpus_dir)
  np.save(f'{corpus_dir}X.npy', X)
  np.save(f'{corpus_dir}y.npy', y)

def save_Xy_data_motion(timesteps=30):
  ''' Compiles all the numpy files for motion into a corpus dir '''
  X, y = create_Xy_data_motion(timesteps=timesteps)
  create_directory_if_needed(MOTION_CORPUS_DIR)
  np.save(f'{MOTION_CORPUS_DIR}X.npy', X)
  np.save(f'{MOTION_CORPUS_DIR}y.npy', y)

def retrieve_Xy_data(corpus_dir=LETTER_CORPUS_DIR):
  ''' Fetch the X and y data from a corpus directory '''
  return np.load(f"{corpus_dir}X.npy"), np.load(f"{corpus_dir}y.npy")

def plot_cm(y_pred, y_test, categories=LETTERS):
  ''' Plot a confusion matrix from predictions and actual labels '''
  cm = confusion_matrix(y_test, y_pred)
  make_confusion_matrix(cm, categories=categories, percent=False)
  plt.show()

def get_one_hot(targets, n_classes):
  ''' One-hot encodes a target variable '''
  res = np.eye(n_classes)[np.array(targets).reshape(-1)]
  return res.reshape(list(targets.shape)+[n_classes])

def get_hand_angle(hand, verbose=False):
  ''' Finds the angle of the hand with respect to facing forward, using the plane of the palm of the hand 
      This returns a tuple of three angles, in radians:  
        around_up_angle: angle to rotate around UP axis
        around_right_angle: angle to rotate around RIGHT axis  
        around_out_angle: angle to rotate around OUT axis'''
  base_of_thumb, index_knuckle, pinkie_knuckle = hand[1], hand[5], hand[17]
  # Get vector normal to the plane formed by the palm
  palm_vector = get_normal_from_three_pts(base_of_thumb, pinkie_knuckle, index_knuckle)

  if verbose:
    print("")
    print(f"Base of thumb: {base_of_thumb}\nPinkie knuckle: {pinkie_knuckle}\nIndex knuckle: {index_knuckle}")
    print(f"Palm vector: {palm_vector}")

  # Get XZ (flat, like on a table) projection of palm vector
  xz_projection = project_vector_onto_plane(palm_vector, UP)
  # Get angle required to rotate the hand vector around the UP vector to be "in front of us"
  around_up_angle = angle_between_360(xz_projection, OUT, UP)
  # Rotate normal vector by that amount
  forward_vector = rotate_around_up(palm_vector, around_up_angle)
  if verbose:
    print(f"Around up angle: {around_up_angle} radians")
    print(f"Forward-facing vector: {forward_vector}")

  # Get angle required to rotate hand up / down so it is aligned with "out"
  # After this rotation, the palm is facing completely out
  around_right_angle = angle_between_360(forward_vector, OUT, RIGHT)
  aligned_vector = rotate_around_right(forward_vector, around_right_angle)
  if verbose:
    print(f"Up-down angle: {around_right_angle} radians")
    print(f"Plane-aligned vector: {aligned_vector}")
  # Check that the plane is aligned
  if not np.allclose(aligned_vector, np.array([0, 0, 1])):
    print(f"[WARNING]: PALM NORMAL VECTOR DID NOT ALIGN WITH PLANE: {aligned_vector}")

  # Now that we can align the palm of the hand, let's find the angle needed to "wave" the hand to point up
  hand_vector = index_knuckle - base_of_thumb
  hand_vector = rotate_around_up(hand_vector, around_up_angle)
  hand_vector = rotate_around_right(hand_vector, around_right_angle)
  around_out_angle = angle_between_360(hand_vector, UP, OUT)
  if verbose:
    print(f"Around out angle: {around_out_angle} radians")

  check_alignment = rotate_around_out(hand_vector, around_out_angle)
  # Check that the hand is facing up
  if not np.allclose(check_alignment[0], 0) or not np.allclose(check_alignment[2], 0):
    print(f"[WARNING]: HAND IS NOT FACING UP: {check_alignment}")

  if verbose:
    print("")

  return (around_up_angle, around_right_angle, around_out_angle)

def rotate_vector(vector, around_up_angle, around_right_angle, around_out_angle):
  ''' Rotate a vector with two angles (see get_hand_angle) '''
  # Rotate left-right first
  new_vec = vector.copy()
  new_vec = rotate_around_up(new_vec, around_up_angle)
  new_vec = rotate_around_right(new_vec, around_right_angle)
  new_vec = rotate_around_out(new_vec, around_out_angle)
  return new_vec

def normalize(hand, size=True, angle=False):
  ''' Normalize a hand. Options for normalizing size and angle  
      Returns (normalized hand, angle in tuple format)'''
  hand = hand.copy()
  # hand[:, 1] *= ratio
  angles = (0, 0, 0)
  if angle:
    hand, angles = normalize_hand_angle(hand)
  if size:
    hand = normalize_size(hand)
  return hand, angles

def normalize_hand_angle(hand):
  ''' Rotates a hand such that the vector between points 1 and 5 are aligned with the vertical  
      Returns (new hand, the angles in tuple format)'''
  angles = get_hand_angle(hand)
  around_up_angle, around_right_angle, around_out_angle = angles
  new_hand = np.zeros(hand.shape)
  for i, landmark in enumerate(hand):
    new_hand[i] = rotate_vector(landmark, around_up_angle, around_right_angle, around_out_angle)
  return new_hand, angles

def create_directory_if_needed(dirname, verbose=False):
  ''' Creates a directory if it does not exist. If it does exist, nothing happens '''
  if os.path.isdir(dirname):
    if verbose:
      print(f"[create_directory_if_needed] Directory '{dirname}' already exists. Aborting.")
    return
  os.makedirs(dirname)
  if verbose:
    print(f"[create_directory_if_needed] Directory '{dirname}' created.")

def normalize_directory(directory, new_directory='NORMALIZED_DEFAULT_DIR/', size=True, angle=False):
  ''' Angle-normalizes and saves all np files in directory to new_directory '''
  if directory == new_directory:
    print("[angle_normalize_directory]: WARNING: You are replacing existing files")
  create_directory_if_needed(new_directory)
  items = os.listdir(directory)
  # Get all np files
  np_files = [i for i in items if i[-4:] == '.npy']
  # Traverse through np files and load
  for file in np_files:
    hands = np.load(f"{directory}{file}")
    # Iterate through each hand in file
    for i, hand in enumerate(hands):
      # Angle normalize and replace existing hand
      normalized_hand, _ = normalize(hand, size=size, angle=angle)
      hands[i] = normalized_hand
    # Replace filename with new sign file
    np.save(f"{new_directory}{file}", hands)

def plot_hand(hand):
  ''' Plots a hand (21, 3) in 3D space '''

  traces = {'thumb': list(range(5)),
            'index': list(range(5,9)),
            'mid': list(range(9,13)),
            'ring': list(range(13,17)),
            'pinky': [0] + list(range(17,21)),
            'palm': [0] + list(range(5,18,4))}
  trace_filter = lambda x: [True if i in traces[x] else False for i in range(21)]

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(hand[:,0], hand[:,1], hand[:, 2], marker='o')
  ax.set_xlabel('X (RIGHT)')
  ax.set_ylabel('Y (UP)')
  ax.set_zlabel('Z (OUT)')
  ax.set_xlim([-1,1])
  ax.set_ylim([-1,1])
  ax.set_zlim([-1,1 ])
  for section in traces:
    x, y, z = hand[trace_filter(section)][:,0], hand[trace_filter(section)][:,1], hand[trace_filter(section)][:,2]   
    ax.plot(x, y, z, color = 'b')
  plt.show()

def check_normalized(directory, size=True, angle=False, verbose=False):
  ''' Checks if a directory is normalized, with options for size and angle  
      Returns true if every hand is normalized, false otherwise '''
  items = os.listdir(directory)
  # Get all np files
  np_files = [i for i in items if i[-4:] == '.npy']
  # Traverse through np files and load
  for file in tqdm(np_files):
    hands = np.load(f"{directory}{file}")
    for i, hand in enumerate(hands):
      if size:
        maximum = 0
        for vec in hand:
          maximum = max(maximum, np.linalg.norm(vec))
        if not np.allclose(maximum, 1.0):
          if verbose:
            print(f'[WARNING]: Maximum vector length not 1 at index {i} of file {file}')
            plot_hand(hand)
          return False
      if angle:
        palm_vector = get_normal_from_three_pts(hand[1], hand[17], hand[5])
        up_vector = hand[5] - hand[1]
        if not np.allclose(palm_vector, OUT):
          if verbose:
            print(f'[WARNING]: Hand not facing outwards at index {i} of file {file}')
            plot_hand(hand)
          return False
        if not np.allclose(unit_vector(up_vector), UP):
          if verbose:
            print(f'[WARNING]: Hand not aligned with the vertical at index {i} of file {file}')
            plot_hand(hand)
          return False
  return True

def remove_large_values(data, threshold=5):
  to_remove = []
  for i, sample in enumerate(data):
    print(np.max(sample))
    print(np.min(sample))
    if np.max(sample) > threshold or np.min(sample) < -threshold:
      to_remove.append(i)
  print(to_remove)
  data = [sample for i in range(len(data)) if i not in to_remove]
  return np.array(data)