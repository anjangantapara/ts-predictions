""" Example code for fitting an RNN to a univariate time series"""
import numpy as np
from matplotlib import pyplot as plt
from nptyping import NDArray
from tensorflow import keras

from ts_data_prep import generate_time_series


def last_time_step_mse(Y_true: NDArray, Y_pred: NDArray) -> float:
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])


n_steps = 50
# generating dummy uni-variate time series data
series = generate_time_series(10000, n_steps + 10)

# preparing X or input data for the time series models
X_train = series[:7000, :n_steps]
X_valid = series[7000:9000, :n_steps]
X_test = series[9000:, :n_steps]

# preparing time series target data for a sequence to sequence to models
# creating 10 day target vectors
Y = np.empty((10000, n_steps, 10))  # each target is a sequence of 10D vectors
for step_ahead in range(1, 10 + 1):
    Y[:, :, step_ahead - 1] = series[:, step_ahead:step_ahead + n_steps, 0]
Y_train = Y[:7000]
Y_valid = Y[7000:9000]
Y_test = Y[9000:]

# Seq-Seq model building
model = keras.models.Sequential([
    keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.LSTM(20, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(10))
])
model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
history = model.fit(X_train, Y_train, epochs=20, validation_data=(X_valid, Y_valid))

# store model to disc
model.save("../data/model_artifacts/" + "simpleLSTM.model")

Y_pred = model.predict(X_test)

plt.plot(series[9000, :], '.')
plt.plot(Y_test[0, :, 0], 'g*')
plt.plot(Y_pred[0, :, 0], 'ro')
plt.show()
