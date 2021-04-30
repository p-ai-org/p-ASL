import cv2
import numpy as np
import mediapipe as mp
from joblib import dump, load
import math
from utilities.util import *
import utilities.cv2utils as cv2utils
mp_drawing = mp.solutions.drawing_utils

mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

# Constants
DATASET_SIZE = 12
FRAMES_PER_SAMPLE = 30
# What to name this numpy file
FNAME = 'L_ABOVE_FIVE'
print(FNAME)
# How many frames to give to "reset"
FRAMES_PER_RESET = 40
# Get screen ratio
RATIO = cap.get(4) / cap.get(3)

''' Where to save this data '''
SAVE_DIR = HOLISTIC_DIR 

DATASET_DIMS = (FRAMES_PER_SAMPLE, ((NUM_DIM - 1) * 2) + (NUM_DIM * NUM_POINTS * 2))

# Load MediaPipe model
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# What will end up being the dataset collected during this session
dataset = np.empty((1, DATASET_DIMS[0], DATASET_DIMS[1]))

count = 0
reset_count = 0
reset_period = False

# Add one to timesteps because last one is always messed up
this_sample = np.empty((DATASET_DIMS[0] + 1, DATASET_DIMS[1]))

# bools for ensuring we don't start collecting data until hands are found
left_found = False
right_found = False

while cap.isOpened():
  ''' Reading frame '''
  success, image = cap.read()
  if not success:
    print("Ignoring empty camera frame.")
    continue

  # Give you time to reset your hands
  if reset_period:
    if reset_count == 0:
      print('[reset]')
    reset_count += 1
    if reset_count == FRAMES_PER_RESET:
      reset_count = 0
      reset_period = False
      print('>> RECORDING')
    continue

  # Process and get hands
  image, results = cv2utils.process_and_identify_landmarks(image, holistic)

  # Draw landmark annotation on the image.
  mp_drawing.draw_landmarks(
      image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
  mp_drawing.draw_landmarks(
      image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
  mp_drawing.draw_landmarks(
      image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
  mp_drawing.draw_landmarks(
      image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

  # Finding hand landmarks for the first time
  if (not right_found) and results.right_hand_landmarks:
    right_found = True

  if (not left_found) and results.left_hand_landmarks:
    left_found = True    
    
  # If we found pose
  if results.pose_landmarks and right_found and left_found:
    # Find position of hands
    pose_np = pose_landmarks_to_np(results.pose_landmarks.landmark)
    left_hand_pos, right_hand_pos = pose_np[19], pose_np[20]
    
    # Discard the z coordinate (model is not fully trained to predict depth)
    left_hand_pos = left_hand_pos[:-1]
    right_hand_pos = right_hand_pos[:-1]
    
    # Updating hand landmarks if they are found again
    if results.right_hand_landmarks:
      right_hand_points = landmarks_to_np(results.right_hand_landmarks.landmark)
      right_hand_points, _ = normalize_hand(right_hand_points, screenRatio=RATIO)
      right_hand_points = right_hand_points.flatten()
    
    if results.left_hand_landmarks:
      left_hand_points = landmarks_to_np(results.left_hand_landmarks.landmark)
      left_hand_points, _ = normalize_hand(left_hand_points, screenRatio=RATIO, leftHand=True)
      left_hand_points = left_hand_points.flatten()
    
    torso_width = (pose_np[11] - pose_np[12])[0]
    torso_height = (pose_np[23] - pose_np[11])[1]

    # We'll set the origin to be in the middle of the right and left shoulder
    # This is better than having (0, 0) be at the top left of the screen
    origin = np.array([torso_width / 2 + pose_np[12][0], pose_np[12][1]])
    left_hand_pos -= origin
    right_hand_pos -= origin

    # Scale according to torso size (to account for distance to camera)
    left_hand_pos = [left_hand_pos[0] / torso_width, left_hand_pos[1] / torso_height]
    right_hand_pos = [right_hand_pos[0] / torso_width, right_hand_pos[1] / torso_height]
    

    ''' Printing stuff'''
    origin_text = f"Origin: {[round(x, 3) for x in origin]}"
    image = cv2utils.add_text(image, text=origin_text, right=50, top=50, size=0.75, color=(0,255,0), thickness=2)
    hand_text = f"Left: {[round(x, 3) for x in left_hand_pos]}, Right: {[round(x, 3) for x in right_hand_pos]}"
    image = cv2utils.add_text(image, text=hand_text, right=50, top=100, size=0.75, color=(0,0,255), thickness=2)
    
    # Add dataset count
    image = cv2utils.add_text(image, text=str(dataset.shape[0]-1), right=50, top=500, size=3, color=(255,255,0), thickness=3)

    timestep = np.concatenate((left_hand_pos, right_hand_pos, right_hand_points, left_hand_points))
    
    # Add this timestep to the current sample
    this_sample[count] = timestep
    

    count += 1
    # If we've finished this sample
    if count == FRAMES_PER_SAMPLE + 1:
      count = 0
      print(dataset.shape)
      print(this_sample.shape)

      #this part!!
      dataset = np.concatenate((dataset, [this_sample[:-1]]))
      print(f'>> RECORDING CAPTURED ({dataset.shape[0] - 1})')
      reset_period = True

    # If reached desired size, finish up
    if dataset.shape[0] - 1 == DATASET_SIZE:
      # First one was all zeros
      dataset = dataset[1:]
      break
  cv2.imshow('Trainer', image)
  if cv2.waitKey(5) & 0xFF == 27:
    break
holistic.close()
cap.release()

''' Save database '''
create_directory_if_needed(SAVE_DIR)
np.save(f'{SAVE_DIR}{FNAME}.npy', dataset)
