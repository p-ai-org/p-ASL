import cv2

def process_and_identify_hands(image, hands):
  ''' Processes an image and identifies the hands in it.  
      Takes in an image from the videcapture and the mediapipe hands classifier'''
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
  return image, results

def add_text(image, text="", right=0, top=0, size=1, color=(255,0,0), thickness=1):
  ''' Adds text to an image using cv2 '''
  return cv2.putText(
    image, 
    str(text),
    (right, top), 
    cv2.FONT_HERSHEY_SIMPLEX,  
    size,
    color,
    thickness,
    cv2.LINE_AA)