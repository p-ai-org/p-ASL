import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from letters import CLASS_TO_LETTER
from joblib import dump, load
import math
from util import *
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Constants
DATASET_SIZE = 2000
SHUTTER_TIME = 1 * FPS

cap = cv2.VideoCapture(0)
dataset = np.empty((1, NUM_POINTS, NUM_DIM))
done = False
fname = 'Q'
hands = mp_hands.Hands(
  min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)
shutter = False
ticker = 0

while cap.isOpened():
  success, image = cap.read()
  if not success:
    print("Ignoring empty camera frame.")
    # If loading a video, use 'break' instead of 'continue'.
    continue

  if shutter and ticker < SHUTTER_TIME:
    ticker += 1
    continue

  # Flip the image horizontally for a later selfie-view display, and convert
  # the BGR image to RGB.
  image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
  # To improve performance, optionally mark the image as not writeable to
  # pass by reference.
  image.flags.writeable = False
  results = hands.process(image)
  # Draw the hand annotations on the image.
  image.flags.writeable = True
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
      dataset = np.concatenate((dataset, landmarks_to_np(hand_landmarks.landmark)))
      print(dataset.shape[0] - 1)
      if dataset.shape[0] - 1 == DATASET_SIZE:
        dataset = dataset[1:]
        done = True
        break
      mp_drawing.draw_landmarks(
        image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=6, circle_radius=3),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255)),
        color_fn=z_color_fn)
    if done:
      break
  cv2.imshow('Trainer', image)
  if cv2.waitKey(5) & 0xFF == 27:
    break
  ticker = 0
hands.close()
cap.release()
np.save('{}{}.npy'.format(LETTER_DATA_DIR, fname), dataset)
