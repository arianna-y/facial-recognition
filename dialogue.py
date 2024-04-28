from imp import load_module
import numpy as np
import pandas as pd
import csv
from tensorflow import keras 
from keras.utils import to_categorical 
from keras.models import Module 
from keras.layers import Input, LSTM, Dense 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Load data 
data = pd.read_csv('Conversation.csv')

# Dialogue data
input_texts = data['question'].tolist()
target_texts = data['answer'].tolist()
# input_texts = [
#     "How are you?",
#     "What are you doing?",
#     "What social situations do you find challenging or want to work on?",
#     "Do you have any specific goals for improving your social skills?",
#     "Any interesting encounters or observations you'd like to discuss?",

# ]

# target_texts = [
#     "I am good!",
#     "I am chatting with you.",
# ]

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(input_texts + target_texts)
num_encoder_tokens = len(tokenizer.word_index) + 1

input_sequences = tokenizer.texts_to_sequence(input_texts)
target_sequences = tokenizer.texts_to_sequences(target_texts)

max_seq_length = max(max(len(seq) for seq in input_sequences),
                     max(len(seq) for seq in target_sequences))

# Padding sequences
encoder_input_data = pad_sequences(input_sequences, maxlen=max_seq_length, padding='post')
decoder_input_data = pad_sequences(target_sequences, maxlen=max_seq_length, padding='post')

# Define model architecture
latent_dim = 256

encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(num_encoder_tokens, latent_dim) (encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(num_encoder_tokens, latent_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(num_encoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = load_module('Conversation.csv')
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
epochs = 100
batch_size = 64
checkpoint_callback = ModelCheckpoint('Conversation.csv', monitor='val_loss', save_best_only=True, verbose=1)

model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          callbacks=[checkpoint_callback])

# Generate response using the trained model
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index['']