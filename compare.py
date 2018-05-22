import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from logistic_function import *

#compare resolving power of logistic function and convolution at different step amplitude
step = np.around(np.exp(np.linspace(0, 2, 25)) - 1, decimals = 1)
num_case_per_amp = 1
center = 1.5
x = np.linspace(1,2,1000)

ws_convolve = 50
flt = np.concatenate((np.ones(ws_convolve), np.ones(ws_convolve) * -1), axis = 0) / ws_convolve

result_logistic_func = np.zeros((len(step), num_case_per_amp, 4))
result_convolution = np.zeros((len(step), num_case_per_amp, 2))

for i in range(len(step)):#set amplitude
    for j in range(num_case_per_amp):#100 test case for every amplitude
        np.random.seed(100 * i + j)
        y = np.random.randn(1000)
        y[x > center] += step[i]
        #prepare initial value for logistic function
        initial_values = np.zeros(4)
        initial_values[0] = np.mean(y)
        initial_values[1] = max(y) - min(y)
        initial_values[2] = 50
        initial_values[3] = np.mean(x)
        #fit with logistic function
        try:
            c, delta, k, t0 = curve_fit(logistic, x, y, initial_values)[0]
        except RuntimeError:
            c, delta, k, t0 = np.nan, np.nan, np.nan, np.nan 
        #filter out data points that do not meet criteria (see function documentation)
        if ((t0 > x[-1]) | (t0 < x[0]) | (delta > max(y) - min(y))):
            c, delta, k, t0 = np.nan, np.nan, np.nan, np.nan
        result_logistic_func[i][j] = c, delta, k, t0
        #convolution
        convolved = scipy.signal.convolve(y, flt, mode = "valid")
        result_convolution[i][j] = x[ws_convolve: -ws_convolve + 1][np.nanargmax(convolved)],max(convolved)

np.save("/Users/hdqeing/masterarbeit/master_goe/data/result_logistic_function", result_logistic_func)
np.save("/Users/hdqeing/masterarbeit/master_goe/data/result_convolution", result_convolution)
        

