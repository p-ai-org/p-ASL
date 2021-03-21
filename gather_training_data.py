import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from joblib import dump, load
import math
from utilities.util import *
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

# Constants
DATASET_SIZE = 1000
SHUTTER_TIME = 1 * cap.get(cv2.CAP_PROP_FPS)
SHUTTER = False
# What to name this numpy file
FNAME = 'ONE'

NORMALIZE_ANGLE = False

''' Where to save this data '''
# SAVE_DIR = LETTER_DATA_DIR
SAVE_DIR = CLASSIFIER_DATA_DIR
# SAVE_DIR = CLASSIFIER_NORM_DATA_DIR

# What will end up being the dataset collected during this session
dataset = np.empty((1, NUM_POINTS, NUM_DIM))
done = False
# Load MediaPipe model
hands = mp_hands.Hands(
  min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)
ticker = 0

while cap.isOpened():
  ''' Reading frame '''
  success, image = cap.read()
  if not success:
    print("Ignoring empty camera frame.")
    # If loading a video, use 'break' instead of 'continue'.
    continue
  if SHUTTER and ticker < SHUTTER_TIME:
    ticker += 1
    continue

  ''' CV2 fun '''
  # Flip the image horizontally for a later selfie-view display, and convert
  # the BGR image to RGB.
  image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
  # To improve performance, optionally mark the image as not writeable to
  # pass by reference.
  image.flags.writeable = False

  ''' MediaPipe identifies hand(s) '''
  results = hands.process(image)
  # Draw the hand annotations on the image.
  image.flags.writeable = True
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  
  # If we found hands
  if results.multi_hand_landmarks:
    # For each hand
    for hand_landmarks in results.multi_hand_landmarks:
      # Get landmarks in np format
      hand_np_raw = landmarks_to_np(hand_landmarks.landmark)
      # Normalize size (and angle, if desired)
      hand_np, _ = normalize(hand_np_raw, size=True, angle=NORMALIZE_ANGLE)
      # Concatenate all the hand landmarks to the dataset
      dataset = np.concatenate((dataset, hand_np))
      # Print the current size of the dataset
      print(dataset.shape[0] - 1)
      # If reached desired size, finish up
      if dataset.shape[0] - 1 == DATASET_SIZE:
        dataset = dataset[1:]
        done = True
        break
      # Draw hand landmarks
      mp_drawing.draw_landmarks(
        image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=6, circle_radius=3),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255)))
    if done:
      break
  cv2.imshow('Trainer', image)
  if cv2.waitKey(5) & 0xFF == 27:
    break
  ticker = 0
hands.close()
cap.release()

''' Save database '''
create_directory_if_needed(SAVE_DIR)
np.save(f'{SAVE_DIR}{FNAME}.npy', dataset)
