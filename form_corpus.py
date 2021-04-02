from utilities.util import *

name_directory_corpus = [
  (LETTERS, LETTER_DATA_DIR, LETTER_CORPUS_DIR),
  (CLASSIFIERS, CLASSIFIER_DATA_DIR, CLASSIFIER_CORPUS_DIR),
  (CLASSIFIERS, CLASSIFIER_NORM_DATA_DIR, CLASSIFIER_NORM_CORPUS_DIR)
]

# Hand data
for name, directory, corpus in name_directory_corpus:
  print(f"Saving {directory} to {corpus}")
  save_Xy_data(corpus_dir=corpus, names=name, data_dir=directory)

# Motion data
print("Saving motion data")
save_Xy_data_motion(timesteps=30)