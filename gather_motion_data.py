import cv2
import numpy as np
import mediapipe as mp
from joblib import dump, load
import math
from utilities.util import *
import utilities.cv2utils as cv2utils
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

# Constants
DATASET_SIZE = 50
FRAMES_PER_SAMPLE = 30
# What to name this numpy file
FNAME = 'MOVE_APART'
# How many frames to give to "reset"
FRAMES_PER_RESET = 30

''' Where to save this data '''
SAVE_DIR = MOTION_DATA_DIR

# Load MediaPipe model
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, upper_body_only=False)
# What will end up being the dataset collected during this session
dataset = np.empty((1, FRAMES_PER_SAMPLE, NUM_DIM * 2))
count = 0
reset_count = 0
reset_period = False

this_sample = np.empty((FRAMES_PER_SAMPLE + 1, NUM_DIM * 2))

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
  image, results = cv2utils.process_and_identify_landmarks(image, pose)
  
  # Draw the pose annotation on the image.
  image.flags.writeable = True
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  mp_drawing.draw_landmarks(
      image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

  # If we found hands
  if results.pose_landmarks:
    # Get landmarks in np format
    pose_np = pose_landmarks_to_np(results.pose_landmarks.landmark)
    
    left_hand, right_hand = pose_np[19], pose_np[20]
    
    torso_width = (pose_np[11] - pose_np[12])[0]
    torso_height = (pose_np[23] - pose_np[11])[1]

    # We'll set the origin to be in the middle of the right and left shoulder
    # This is better than having (0, 0) be at the top left of the screen
    origin = np.array([torso_width / 2 + pose_np[12][0], pose_np[12][1], 0])
    left_hand -= origin
    right_hand -= origin

    # Scale according to torso size (to account for distance to camera)
    left_hand = [left_hand[0] / torso_width, left_hand[1] / torso_height, left_hand[2] / torso_width]
    right_hand = [right_hand[0] / torso_width, right_hand[1] / torso_height, right_hand[2] / torso_width]

    ''' Printing stuff '''
    origin_text = f"Origin: {[round(x, 3) for x in origin]}"
    image = cv2utils.add_text(image, text=origin_text, right=50, top=50, size=0.75, color=(0,255,0), thickness=2)
    hand_text = f"Left: {[round(x, 3) for x in left_hand]}, Right: {[round(x, 3) for x in right_hand]}"
    image = cv2utils.add_text(image, text=hand_text, right=50, top=100, size=0.75, color=(0,0,255), thickness=2)

    # Add dataset count
    image = cv2utils.add_text(image, text=str(dataset.shape[0]-1), right=50, top=500, size=3, color=(255,255,0), thickness=3)

    hands_data = np.concatenate((left_hand, right_hand))

    # Add this timestep to the current sample
    this_sample[count] = hands_data
    count += 1
    # If we've finished this sample
    if count == FRAMES_PER_SAMPLE + 1:
      count = 0
      print(dataset.shape)
      print(this_sample.shape)
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
pose.close()
cap.release()

''' Save database '''
create_directory_if_needed(SAVE_DIR)
np.save(f'{SAVE_DIR}{FNAME}.npy', dataset)
