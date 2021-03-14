import numpy as np
from nptyping import NDArray
from numpy import array


def generate_time_series(batch_size: int, n_steps: int) -> NDArray:
    """ Function to generate time series
    From Aurelien Geron book. Chapter 15"""
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  # wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20))  # +wave 2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)  # +noise
    return series[..., np.newaxis].astype(np.float32)


def generate_time_series_seq(batch_size: int, n_steps: int) -> NDArray:
    """ Function to generate time series
    From Aurelien Geron book. Chapter 15"""
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  # wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20))  # +wave 2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)  # +noise
    return series.astype(np.float32)


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
    # todo update this function to prepare the data, to fit to the current needs of multi-dimensional array
    X, y = [], []
    total_time_steps = sequences.shape[1]
    for i in range(total_time_steps):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix > total_time_steps:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[:, i:end_ix], sequences[:, end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# # split a multivariate sequence into samples
# def split_sequences(sequences, n_steps_in, n_steps_out):
#     # todo update this function to prepare the data, to fit to the current needs of multi-dimensional array
#     X, y = np.empty((1000, n_steps_in, 1)), np.empty((1000, n_steps_out, 1))
#     print("seq",sequences.shape)
#     for i in range(len(sequences)):
#         # find the end of this pattern
#         end_ix = i + n_steps_in
#         out_end_ix = end_ix + n_steps_out
#         # check if we are beyond the dataset
#         if out_end_ix > len(sequences):
#             break
#         # gather input and output parts of the pattern
#         seq_x, seq_y = sequences[:, i:end_ix], sequences[:, end_ix:out_end_ix]
#         X[i, :, :] = seq_x
#         y[i, :, :] = (seq_y)
#     return X, y
