import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from utilities.reference import *
from joblib import dump, load
import math
from utilities.util import *
import utilities.cv2utils as cv2utils
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# How confident the prediction needs to be
CONFIDENCE_THRESHOLD = 0
# How long to be predicting the same thing before "officially" predicting it
TIME_THRESHOLD_IN_SEC = 0

# Which model to use
MODEL_NAME = 'nn_non_norm'
# Set to true if model is a SVM
USE_SVM = False
# Whether to normalize the hand to be facing outward (in theory)
NORMALIZE_ANGLE = True

# Show the landmarks in the video
SHOW_HAND = True

cap = cv2.VideoCapture(0)
consistency_counter = 0
hands = mp_hands.Hands(
  min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)
last_pred = None
time_threshold_in_frames = TIME_THRESHOLD_IN_SEC * cap.get(cv2.CAP_PROP_FPS)

if USE_SVM:
  recognizer = load(f"{MODEL_DIR}{MODEL_NAME}")
else:
  recognizer = tf.keras.models.load_model(f"{MODEL_DIR}{MODEL_NAME}")

def classify(hand_np):
  ''' Classifies a hand, given the hand in (21, 3)  
      Returns probability distribution, most likely index, and the corresponding classifier'''
  # Flatten to (1, 63)
  hand_np_flat = np.array([flatten_hand(hand_np)])
  # Get probability distribution with classifier
  if USE_SVM:
    probs = recognizer.predict_proba(hand_np_flat)[0]
  else:
    probs = recognizer.predict(hand_np_flat)[0]
  pred_index = np.argmax(probs)
  classifier_pred = index_to_classifier(int(round(pred_index)))
  return probs, pred_index, classifier_pred

while cap.isOpened():
  success, image = cap.read()
  
  if not success:
    print("Ignoring empty camera frame.")
    continue

  # Process and get hands
  image, results = cv2utils.process_and_identify_hands(image, hands)

  if results.multi_hand_landmarks:
    # For each hand (not relevant yet)
    for hand_landmarks in results.multi_hand_landmarks:
      # Turn landmarks into a (21, 3) np array
      hand_np_raw = landmarks_to_np(hand_landmarks.landmark)
      # Apply normalization
      hand_np, angles = normalize(hand_np_raw, size=True, angle=NORMALIZE_ANGLE)

      angle_text = f"Rotation (up, right, out): {[round(x, 3) for x in angles]}"
      image = cv2utils.add_text(image, text=angle_text, right=100, top=200, color=(255, 255, 255), size=0.5)
      
      hand_center_text = f"Center (x, y, z): {[round(x, 3) for x in get_center_of_hand(hand_np_raw)]}"
      image = cv2utils.add_text(image, text=hand_center_text, right=100, top=150, size=0.5)

      # Get probability distribution, predicted index, and corresponding classifier
      probs, pred_index, classifier_pred = classify(hand_np)

      # Build confidence in this prediction (not unstable)
      if classifier_pred == last_pred:
        consistency_counter += 1
      
      # If prediction is confident enough and not the last confirmed letter (CHANGE)
      if probs[pred_index] >= CONFIDENCE_THRESHOLD:
        # If consistent enough
        if consistency_counter >= time_threshold_in_frames:
          # Show prediction
          image = cv2utils.add_text(image, text=classifier_pred, right=100, top=100, size=3, color=(0, 255, 0), thickness=4)
      # Not a confident prediction
      if probs[pred_index] < CONFIDENCE_THRESHOLD or classifier_pred != last_pred:
        consistency_counter = 0
      last_pred = classifier_pred
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