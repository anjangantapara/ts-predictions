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
