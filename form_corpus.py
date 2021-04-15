from utilities.util import *

name_directory = [
  (LETTERS, LETTER_DIR),
  (CLASSIFIERS, CLASSIFIER_ANYANGLE_DIR),
  (CLASSIFIERS, CLASSIFIER_FORCED_DIR),
  (CLASSIFIERS, CLASSIFIER_UPRIGHT_DIR)
]

# Hand data
for name, directory in name_directory:
  print(f"Saving {directory}")
  save_Xy_data(names=name, data_dir=directory)

# Motion data
print("Saving motion data")
save_Xy_data_motion(names=MOTIONS, data_dir=MOTION_DIR, timesteps=30)