import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from utilities.reference import *
from joblib import dump, load
import math
import cv2utils
from utilities.util import *
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

CONFIDENCE_THRESHOLD = 0.9
RESET_LENGTH = 0.05
TIME_THRESHOLD_IN_SEC = 0.3

MODEL_NAME = 'fingerspelling_nn'
USE_SVM = False

SHOW_HAND = True

build_str = ''
cap = cv2.VideoCapture(0)
consistency_counter = 0
half_reset = False
reset = False
hands = mp_hands.Hands(
  min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)
last_detected_mean = None
last_pred = None
last_letter = None
time_threshold_in_frames = TIME_THRESHOLD_IN_SEC * cap.get(cv2.CAP_PROP_FPS)

if USE_SVM:
  recognizer = load(f"{MODEL_DIR}{MODEL_NAME}")
else:
  recognizer = tf.keras.models.load_model(f"{MODEL_DIR}{MODEL_NAME}")

def classify(hand_np):
  ''' Classifies a hand, given the hand in (21, 3)  
      Returns probability distribution, most likely index, and the corresponding letter'''
  # Flatten to (1, 63)
  hand_np_flat = np.array([flatten_hand(hand_np)])
  # Get probability distribution with classifier
  if USE_SVM:
    probs = recognizer.predict_proba(hand_np_flat)[0]
  else:
    probs = recognizer.predict(hand_np_flat)[0]
  pred_index = np.argmax(probs)
  letter_pred = index_to_letter(int(round(pred_index)))
  return probs, pred_index, letter_pred

while cap.isOpened():
  success, image = cap.read()
  if not success:
    print("Ignoring empty camera frame.")
    continue

  # Process and get hands
  image, results = cv2utils.process_and_identify_hands(image, hands)

  # Show building string
  image = cv2utils.add_text(image, text=build_str, right=100, top=200, size=2, color=(0, 0, 255), thickness=3)

  if results.multi_hand_landmarks:
    # For each hand (not relevant yet)
    for hand_landmarks in results.multi_hand_landmarks:
      # Shape (21, 3)
      hand_np_raw = landmarks_to_np(hand_landmarks.landmark)
      hand_np, _ = normalize(hand_np_raw, size=True, angle=False)
      
      dist_from_last_detection = euclidian_distance(
        last_detected_mean, 
        get_center_of_hand(hand_np_raw), 
        z_scale=2)
      # print(dist_from_last_detection)
      if dist_from_last_detection and dist_from_last_detection > RESET_LENGTH:
        half_reset = True
      else:
        if half_reset:
          reset = True
      
      probs, pred_index, letter_pred = classify(hand_np)

      # Build confidence in this prediction (not unstable)
      if letter_pred == last_pred:
        consistency_counter += 1
      # If prediction is confident enough and not the last confirmed letter (CHANGE)
      if probs[pred_index] >= CONFIDENCE_THRESHOLD and (last_letter != letter_pred or reset):
        # If consistent enough
        if consistency_counter >= time_threshold_in_frames:
          # Remember where this hand was
          last_detected_mean = get_center_of_hand(hand_np_raw)
          build_str += letter_pred
          last_letter = letter_pred
          reset = False
          half_reset = False
          # Show prediction
          image = cv2utils.add_text(image, letter_pred, right=100, top=100, size=3, color=(0, 255, 0), thickness=4)
          consistency_counter = 0
      # Not a confident prediction
      if probs[pred_index] < CONFIDENCE_THRESHOLD or letter_pred != last_pred:
        consistency_counter = 0
      last_pred = letter_pred
      if SHOW_HAND:
        mp_drawing.draw_landmarks(
          image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
          landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=6, circle_radius=3),
          connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255)))
  cv2.imshow('Prototype', image)
  if cv2.waitKey(5) & 0xFF == 27:
    break
hands.close()
cap.release()