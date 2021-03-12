import matplotlib.pyplot as plt
from tensorflow import keras

from ts_data_prep import generate_time_series

n_steps = 50
series = generate_time_series(10000, n_steps + 1)
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, :n_steps], series[:9000, -1]
print(X_train.shape)

# loading the model from disc
model = keras.models.load_model('../data/model_artifacts/simpleRNN.model', compile=False)
# casting predictions
y_pred = model.predict(X_test)
# plotting single train, test and pred points
plt.plot(X_train[0, :], 'b-*')
plt.plot(y_test[0, :], '.')
plt.plot(y_pred[0, :], 'ro')
plt.show()


n_steps = 50
series = generate_time_series(10000, n_steps + 10)
X_train, Y_train = series[:7000, :n_steps], series[:7000, -10:0]
X_valid, Y_valid = series[7000:9000, :n_steps], series[7000:9000, -10:0]
X_test, Y_test = series[9000:, :n_steps], series[:9000, -10:0]

# creating 10 day target vectors
Y = np.empty((10000, n_steps, 10))  # each target is a sequence of 10D vectors
for step_ahead in range(1, 10 + 1):

    Y[:, :, step_ahead - 1] = series[:, step_ahead:step_ahead + n_steps, 0]
Y_train = Y[:7000]
Y_valid = Y[7000:9000]
Y_test = Y[9000:]

model = keras.models.Sequential([
    keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.LSTM(20, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(10))
])
model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
history = model.fit(X_train, Y_train, epochs=20, validation_data=(X_valid, Y_valid))
model.save("../data/model_artifacts/" + "simpleLSTM.model")

Y_pred = model.predict(X_test)

plt.plot(X_test[0, :], '.')
plt.plot(Y_test[0, :], 'g*')
plt.plot(Y_pred[0, :], 'ro')
plt.show()

