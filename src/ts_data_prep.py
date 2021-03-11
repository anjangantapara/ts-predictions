import numpy as np
from nptyping import NDArray

def generate_time_series(batch_size: int, n_steps: int) -> NDArray:
    """ Function to generate time series
    From Aurelien Geron book. Chapter 15"""
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  # wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20))  # +wave 2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)  # +noise
    return series[..., np.newaxis].astype(np.float32)
