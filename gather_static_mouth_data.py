from cv2 import cv2
import numpy as np
import mediapipe as mp
from joblib import dump, load
import math
from utilities.util import *
import utilities.cv2utils as cv2utils
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

########################## from GATHER STATIC

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(
    thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)
''' Constants '''
DATASET_SIZE = 50
SHUTTER_TIME = 1 * cap.get(cv2.CAP_PROP_FPS)
SHUTTER = False # ?
# Get video dimensions
WIDTH = cap.get(3)
HEIGHT = cap.get(4)
RATIO = HEIGHT / WIDTH
# What to name this numpy file
FNAME = 'testClosedMouth'

''' Constants for MM'''
MM_NUM_DIM = 3
MM_NUM_POINTS = 80

# 61 185 40 39 37 0 267 269 270 409 291 375 321 405 314 17 84 181 91 146 
# 76 184 74 73 72 11 302 303 304 408 306 307 320 404 315 16 85 180 90 77
# 62 183 42 41 38 12 268 271 272 407 292 325 319 403 316 15 86 179 89 96
# 78 191 80 81 82 13 312 311 310 415 308 324 318 402 317 14 87 178 86 95


# [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 
# 76, 184, 74, 73, 72, 11, 302, 303, 304, 408, 306, 307, 320, 404, 315, 16, 85, 180, 90, 77,
# 62, 183, 42, 41, 38, 12, 268, 271, 272, 407, 292, 325, 319, 403, 316, 15, 86, 179, 89, 96,
# 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 86, 95]

''' Where to save this data ''' 
SAVE_DIR = MOUTH_MORPHEME_DIR

# What will end up being the dataset collected during this session
dataset = np.empty((1, MM_NUM_POINTS, MM_NUM_DIM)) #vars come from util.py
done = False

# Load MediaPipe model
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_faces=1)
    #should this have confidence 0.7 like static hand?
ticker = 0

while cap.isOpened():
  ''' Reading frame '''
  success, image = cap.read() # ?
  if not success:
    print("Ignoring empty camera frame.")
    continue
  if SHUTTER and ticker < SHUTTER_TIME:
    ticker += 1
    continue

  # Process and get face
  image, results = cv2utils.process_and_identify_landmarks(image, face_mesh)
  # what is cv2utils?

  #if we found face
  if results.multi_face_landmarks:
    #for each landmark #if max_num_faces=1, then for loop only goes once ()
    for face_landmarks in results.multi_face_landmarks:
        # Get landmarks in np format 
        #note: the function mouth_landmarks_to_np current is not yet written
        face_np = mouth_landmarks_to_np(face_landmarks.landmark) #make sure your head is upright/not tilted
        # Concatenate all the hand landmarks to the dataset
        dataset = np.concatenate((dataset, [face_np]))
        # Print the current size of the dataset
        print(f"[size] : {dataset.shape[0] - 1}")
        # If reached desired size, finish up
        if dataset.shape[0] - 1 == DATASET_SIZE: #getting rid of first frame?
            #dataset size in10, so 10 matricies. done, and want to check when got to that.
            dataset = dataset[1:]
            done = True
            break
        #Draw face landmarks
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACE_CONNECTIONS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)
    if done:
        break
  cv2.imshow('MediaPipe FaceMesh', image)
  if cv2.waitKey(5) & 0xFF == 27:
     break
  ticker = 0
# hands.close() #do we need an equivalent?

# --------------

cap.release()