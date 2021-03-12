""" Code to plot the time series predictions from an LSTM model"""
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from ts_data_prep import generate_time_series

n_steps = 50
series = generate_time_series(10000, n_steps + 10)
X_train, Y_train = series[:7000, :n_steps], series[:7000, -10:0]
X_valid, Y_valid = series[7000:9000, :n_steps], series[7000:9000, -10:0]
X_test, Y_test = series[9000:, :n_steps], series[9000:, -10:0]

# creating 10 day target vectors
Y = np.empty((10000, n_steps, 10))  # each target is a sequence of 10D vectors
for step_ahead in range(1, 10 + 1):
    Y[:, :, step_ahead - 1] = series[:, step_ahead:step_ahead + n_steps, 0]
Y_train = Y[:7000]
Y_valid = Y[7000:9000]
Y_test = Y[9000:]

model = keras.models.load_model("../data/model_artifacts/" + "simpleLSTM.model", compile=False)

Y_pred = model.predict(X_test)
print(Y_pred[0, 49, :])
time_test = np.arange(50, 60, 1)
print(time_test)
plt.plot(series[9000, :], '.')
plt.plot(time_test, Y_test[0, 49, :], 'g*')
plt.plot(time_test, Y_pred[0, 49, :], 'ro')
plt.show()

print(X_test[0, :].shape)
print("++++++++++++++++++++++++++++")
print(X_test[1, :].shape)
print(X_test.shape)
print(Y_test.shape)
print(Y_pred.shape)
