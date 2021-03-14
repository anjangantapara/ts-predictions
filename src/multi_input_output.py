# multivariate multi-step encoder-decoder lstm example
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
from numpy import hstack
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.models import Sequential

from ts_data_prep import split_sequences, generate_time_series_seq

n_steps = 1000
n_features=1
# define input sequence
# below misusing batch_size=2 for two features
in_seq1 = generate_time_series_seq(n_features, n_steps)
print(in_seq1.shape)
in_seq2 = in_seq1 * 0.8 + 1
out_seq = array([in_seq1[i] + in_seq2[i] for i in range(len(in_seq1))])
# convert to [rows, columns] structure
# in_seq1 = in_seq1.reshape((len(in_seq1), n_steps + 10, 1))
# in_seq2 = in_seq2.reshape((len(in_seq2), n_steps + 10, 1))
# out_seq = out_seq.reshape((len(out_seq), n_steps + 10, 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
print("dataset shape", dataset.shape)
# choose a number of time steps
n_steps_in = 20
n_steps_out = 20
# covert into input/output
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
X = X.reshape(X.shape[0], X.shape[2], n_features)
y = y.reshape(y.shape[0], y.shape[2], n_features)
print("y shape", y.shape)
print("X shape", X.shape)

print("X shape", X.shape)
# define model
model = Sequential()
# lstm input should be as follow batchsize, timesteps and feature_size
model.add(LSTM(200, activation='relu', input_shape=(n_steps, n_features)))
model.add(RepeatVector(n_steps_out))
model.add(LSTM(200, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(n_features)))
model.compile(optimizer='adam', loss='mse')
# fit model
history = model.fit(X, y, epochs=20, verbose=2)
# demonstrate prediction
# list all data in history
# todo move model testing and visualization to another .py file
# todo look for the right metrics for lstm extrapolation
# todo look for data generator
test_ind=100
X_test = in_seq1[:, -test_ind:-test_ind+n_steps_out].reshape(1, n_steps_in, n_features)
y_pred = model.predict(X_test)
y_pred = y_pred.reshape(n_features, n_steps_out)
print(y_pred)
print(X_test.shape)
plt.plot(in_seq1[0, :], 'r-.')
#plt.plot(in_seq1[1, :], 'b-.')
temp_time = np.arange(n_steps-test_ind, n_steps-test_ind + n_steps_out, 1)
plt.figure(1)
plt.plot(temp_time, y_pred[0, :], 'b*')
#plt.plot(temp_time, y_pred[1, :], 'b*')
plt.figure(2)
#
print("-------------- history kerys------------------")
print(history.history.keys())
plt.plot(history.history['loss'], '.')
# plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
