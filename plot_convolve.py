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
    for j in range(1):
        y = np.random.randn(1000)
        y[x > 1.5] += step[i]
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        x_smoothed = moving_median(x, 99)
        y_smoothed = moving_median(y, 99)
        ax1.plot(x, y, 'green', label = "original data")
        ax1.plot(x_smoothed, y_smoothed, "red", label = "smoothed curve")
        ax1.set_title("amplitude = "+str(step[i]), size = 20)
        ax1.set_xlabel("x", size= 20)
        ax1.set_ylabel("y", size = 20)
        ax1.legend(fontsize = 20)
        ax1.tick_params(axis = "both", labelsize = 20)
        ax2 = fig.add_subplot(212)
        convolved = np.load("convolve_amp" + str(step[i]) + "_case_" + str(j)+".npy")
        smooth_convolved = np.load("smooth_convolve_amp" + str(step[i]) + "_case_" + str(j)+".npy")
        diff_n = x.shape[0] - convolved.shape[0]
        diff_n_smooth = y_smoothed.shape[0] - smooth_convolved.shape[0]
        ax2.plot(x[0:len(convolved)], convolved,"green", label = "without smoothing")
        ax2.plot(x[0:len(smooth_convolved)], smooth_convolved,"red", label = "smoothed and convolved")
        ax2.set_xlabel("index", size = 20)
        ax2.set_ylabel("convolution", size = 20)
        ax2.tick_params(axis = "both", labelsize = 20)
        ax2.legend(fontsize = 20)
        plt.show()