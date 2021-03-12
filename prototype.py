import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from utilities.letters import CLASS_TO_LETTER
from joblib import dump, load
import math
from utilities.util import *
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

CONFIDENCE_THRESHOLD = 0.9
RESET_LENGTH = 0.05
TIME_THRESHOLD_IN_SEC = 0.3

build_str = ''
cap = cv2.VideoCapture(0)
consistency_counter = 0
half_reset = False
hands = mp_hands.Hands(
  min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)
last_detected_mean = None
last_pred = None
last_letter = None
reset = False
show_hand = True
time_threshold_in_frames = TIME_THRESHOLD_IN_SEC * FPS
use_svm = False

if use_svm:
  recognizer = load('{}recognizer_svm'.format(MODEL_DIR))
else:
  recognizer = tf.keras.models.load_model('{}recognizer_nn'.format(MODEL_DIR))

while cap.isOpened():
  success, image = cap.read()
  if not success:
    print("Ignoring empty camera frame.")
    # If loading a video, use 'break' instead of 'continue'.

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

  # Show building string
  image = cv2.putText(image, build_str, (100, 200), cv2.FONT_HERSHEY_SIMPLEX,  
            2, (0, 0, 255), 3, cv2.LINE_AA)

  if results.multi_hand_landmarks:
    # For each hand (not relevant yet)
    for hand_landmarks in results.multi_hand_landmarks:
      # Shape (1, 21, 3)
      hand_np_raw = landmarks_to_np(hand_landmarks.landmark, apply_normalization=False)
      hand_np = normalize_np_hand(hand_np_raw)
      # Shape (1, 63)
      hand_np_flat = flatten_np_hand(hand_np)
      
      dist_from_last_detection = euclidian_distance(last_detected_mean, get_center_of_np_hand(hand_np_raw), exaggerate_z=True)
      # print(dist_from_last_detection)
      if dist_from_last_detection and dist_from_last_detection > RESET_LENGTH:
        half_reset = True
      else:
        if half_reset:
          reset = True
      
      # Get probability distribution
      if use_svm:
        probs = recognizer.predict_proba(hand_np_flat)[0]
      else:
        probs = recognizer.predict(hand_np_flat)[0]
      pred_index = np.argmax(probs)
      letter_pred = CLASS_TO_LETTER[int(round(pred_index))]
      # Build confidence in this prediction (not unstable)
      if letter_pred == last_pred:
        consistency_counter += 1
      # If prediction is confident enough and not the last confirmed letter (CHANGE)
      if probs[pred_index] >= CONFIDENCE_THRESHOLD and (last_letter != letter_pred or reset):
        # If consistent enough
        if consistency_counter >= time_threshold_in_frames:
          # Remember where this hand was
          last_detected_mean = get_center_of_np_hand(hand_np_raw)
          build_str += letter_pred
          last_letter = letter_pred
          reset = False
          half_reset = False
          # Show prediction
          image = cv2.putText(image, letter_pred, (100, 100), cv2.FONT_HERSHEY_SIMPLEX,  
                    3, (0, 255, 0), 4, cv2.LINE_AA)
          consistency_counter = 0
      # Not a confident prediction
      if probs[pred_index] < CONFIDENCE_THRESHOLD or letter_pred != last_pred:
        consistency_counter = 0
      last_pred = letter_pred
      if show_hand:
        mp_drawing.draw_landmarks(
          image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
          landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=6, circle_radius=3),
          connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255)))
  cv2.imshow('Prototype', image)
  if cv2.waitKey(5) & 0xFF == 27:
    break
hands.close()
cap.release()