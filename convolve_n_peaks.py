import numpy as np
import scipy.stats
import scipy.signal
import matplotlib.pyplot as plt
import sys
 
x = np.linspace(1,2,1000)

center_1 = 1.4
center_2 = 1.5

step_1 = 1.5
step_2 = 3

for i in range(20):
    y = np.random.randn(1000)
    if (center_1 < center_2):
        y[(x > center_1) & (x < center_2)] += step_1
        y[x > center_2] += step_2
    else:
        y[x > step_1] += step_1
        y[(x < center_1) & (x > center_2)] += step_2
    z = np.concatenate((np.ones(50), np.ones(50) * -1), axis = 0) 
    convolved1 = scipy.signal.convolve(y, z, mode = "valid")/ 50
    plt.plot(convolved1, '-g')
    plt.show()