import numpy as np
import keras
from utilities.util import DATA_DIR, CORPUS_SUFFIX, MODEL_DIR, create_directory_if_needed

SAVENAME = 'holistic'
SAVE = True

# Load data
encoder_input_data = np.load(f"{DATA_DIR}holistic_data_synth/{CORPUS_SUFFIX}X.npy")
y = np.load(f"{DATA_DIR}holistic_data_synth/{CORPUS_SUFFIX}y.npy")

batch_size = 32  # Batch size for training.
epochs = 10  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = len(encoder_input_data)  # Number of samples to train on.

# Add 1 new timestep and two new tokens
decoder_input_data = np.zeros((y.shape[0], y.shape[1] + 1, y.shape[2] + 2))
decoder_target_data = np.zeros(decoder_input_data.shape)

# Put y data for all samples, starting at the second timestep, leaving alone start and stop token
decoder_input_data[:, 1:, :-2] = y
# Add start token to beginning of every sample
decoder_input_data[:, 0, -1] = 1

# Put y data for all samples, up to the last timestep, leaving alone start and stop token
decoder_target_data[:, :-1, :-2] = y
# Add end token to end of every sample
decoder_target_data[:, -1, -2] = 1

# returns train, inference_encoder and inference_decoder models
def define_models(n_input, n_output, n_units):
	# define training encoder
	encoder_inputs = keras.layers.Input(shape=(None, n_input))
	encoder = keras.layers.LSTM(n_units, return_state=True)
	encoder_outputs, state_h, state_c = encoder(encoder_inputs)
	encoder_states = [state_h, state_c]
	# define training decoder
	decoder_inputs = keras.layers.Input(shape=(None, n_output))
	decoder_lstm = keras.layers.LSTM(n_units, return_sequences=True, return_state=True)
	decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
	decoder_dense = keras.layers.Dense(n_output, activation='softmax')
	decoder_outputs = decoder_dense(decoder_outputs)
	model = keras.models.Model([encoder_inputs, decoder_inputs], decoder_outputs)
	# define inference encoder
	encoder_model = keras.models.Model(encoder_inputs, encoder_states)
	# define inference decoder
	decoder_state_input_h = keras.layers.Input(shape=(n_units,))
	decoder_state_input_c = keras.layers.Input(shape=(n_units,))
	decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
	decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
	decoder_states = [state_h, state_c]
	decoder_outputs = decoder_dense(decoder_outputs)
	decoder_model = keras.models.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
	# return all models
	return model, encoder_model, decoder_model

model, inf_encoder, inf_decoder = define_models(encoder_input_data.shape[-1], decoder_target_data.shape[-1], latent_dim)

model.summary()

model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2
)

# Save model
if SAVE:
  create_directory_if_needed(f"{MODEL_DIR}{SAVENAME}/")
  model.save(f"{MODEL_DIR}{SAVENAME}/full")
  inf_encoder.save(f"{MODEL_DIR}{SAVENAME}/encoder")
  inf_decoder.save(f"{MODEL_DIR}{SAVENAME}/decoder")
