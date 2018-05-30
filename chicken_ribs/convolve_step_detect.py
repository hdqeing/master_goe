import numpy as np
import scipy.stats
import scipy.signal
import matplotlib.pyplot as plt
import sys

step = np.around(np.exp(np.linspace(0, 2, 25)) - 1, decimals = 1)
repeat = 100
x = np.linspace(1, 2, 1000)

def moving_median(L, window_size):
    N = len(L) - window_size + 1
    y_smoothed = np.zeros(N)
    for i in range(N):
        y_smoothed[i] = np.median(L[i : i + window_size])
    return y_smoothed

np.random.seed(19931229)

for i in range(len(step)):
    for j in range(repeat):
        y = np.random.randn(1000)
        y[x > 1.5] += step[i]
        y_smoothed = moving_median(y, 99)
        z = np.concatenate((np.ones(50), np.ones(50) * -1), axis = 0)
        convolved1 = scipy.signal.convolve(y, z, mode = "valid")
        convolved2 = scipy.signal.convolve(y_smoothed, z, mode = "valid")
        np.save("convolve_amp" + str(step[i]) + "_case_" + str(j), convolved1)
        np.save("smooth_convolve_amp" + str(step[i]) + "_case_" + str(j), convolved2)