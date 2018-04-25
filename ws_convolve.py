import numpy as np
import scipy.stats
import scipy.signal
import matplotlib.pyplot as plt
import sys

ws = np.arange(50, 200, 30)
step = 1
#loc = [411,412,413,414,421,422,423,424,431,432,433,434,441,442,443,444]
x = np.linspace(1, 2, 1000)

def moving_median(L, window_size):
    N = len(L) - window_size + 1
    y_smoothed = np.zeros(N)
    for i in range(N):
        y_smoothed[i] = np.median(L[i : i + window_size])
    return y_smoothed

np.random.seed(19931229)

for i in range(len(ws)):
    y = np.random.randn(1000)
    y[x > 1.5] += step
    z = np.concatenate((np.ones(ws[i]), np.ones(ws[i]) * -1), axis = 0)
    convolved = scipy.signal.convolve(y, z, mode = "valid")
    plt.plot(x[ws[i]-1:-ws[i]], convolved/ws[i],label = "window size:" + str(ws[i]))
    plt.xlabel("x")
    plt.ylabel("amplitude")
    plt.legend()

plt.show()

    