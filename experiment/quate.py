from transformers import AutoTokenizer, TFAutoModel
import pdb
import numpy as np
from tensorflow.keras.layers import  AveragePooling1D, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam, SGD

import tensorflow

from tensorflow.keras.layers import  Dense, Dropout, concatenate,Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Model,Sequential, load_model

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = TFAutoModel.from_pretrained("bert-base-uncased")
inputs = tokenizer("I have no life", return_tensors="tf")
outputs = model(**inputs)

# pdb.set_trace()
# _, outputs_s = np.asarray(outputs.last_hidden_state)
seq_length = outputs.last_hidden_state.shape[1]
features_dim = outputs.last_hidden_state.shape[2]

o_model = Sequential()
o_model.add(BatchNormalization(input_shape = (seq_length, features_dim), name = 'audio_rnn_BN_1'))

# lstm layer
o_model.add(LSTM(64, input_shape=(1, seq_length, features_dim), name  = 'audio_rnn_lstm'))
o_model.add(Activation('relu', name = 'audio_rnn_activation1'))
o_model.add(BatchNormalization(name ='audio_rnn_BN_2'))
o_model.add(Dropout(0.5, name = 'audio_rnn_dropout_1'))

o_model.add(Dense(256, name = 'dense_hidden_layer'))
o_model.add(Activation('relu', name = 'dense_activation'))
o_model.add(BatchNormalization(name = 'dense_BN'))
o_model.add(Dropout(0.5, name = 'dense_dropout'))

o_model.add(Dense(1, activation ='sigmoid' , kernel_initializer ='normal', name ='sigmoid_decision_layer'))

o_model.compile(loss = 'mean_squared_error', metrics = ['accuracy'], optimizer = Adam())

pdb.set_trace()
x = np.reshape(np.asarray(outputs.last_hidden_state), (1, 6, 768))
o_model.fit(
        x,
        np.asarray([0.1341]),
        batch_size=1)

pdb.set_trace()