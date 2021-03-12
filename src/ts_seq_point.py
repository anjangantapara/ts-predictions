""" Example code for fitting an RNN to a univariate time series"""
from matplotlib import pyplot as plt
from nptyping import NDArray
from tensorflow import keras

from ts_data_prep import generate_time_series


def last_time_step_mse(Y_true: NDArray, Y_pred: NDArray) -> float:
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])


n_steps = 50
series = generate_time_series(10000, n_steps + 1)
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, :n_steps], series[:9000, -1]
print(X_train.shape)



model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20, return_sequences=True),
    keras.layers.SimpleRNN(1)
])
model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
model.save("../data/model_artifacts/" + "simpleRNN.model")

y_pred = model.predict(X_test)

plt.plot(X_train[0, :], 'b-*')
plt.plot(y_test[0, :], '.')
plt.plot(y_pred[0, :], 'ro')
plt.show()
