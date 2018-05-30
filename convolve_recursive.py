import numpy as np
import scipy.stats
import scipy.signal
import matplotlib.pyplot as plt
import sys

num_dps = 1000

x = np.linspace(1,2,1000)

center_candidate = []

x = np.linspace(1,2,1000)

num_steps = 3 #number of steps

step = 3

repeat = 10

center = []
amplitude = []

def find_peaks(x, y, ws, threshold, mode = 'max'):
    '''
    This function find local extrema in a curve.

    Procedure:
        1. find global maximum/minimum
        2. remove the ws-neighbourhood
        3. find maximum
        4. iterate until the find maximum is smaller than a threshold

    Arguement:
    x: convolved x value;
    y: convolved y value;
    ws: window size for convolving;
    threshold: 
        in 'max' mode, if find amplitude is smaller than threshold, the process is ended;
        in 'min' mode, if find amplitude is larger than threshold, the process is ended.
    mode: 'max' or 'min', optional, max default. Search for local maximum or local minimum. 
    '''
    if (len(y) == 0):
        return
    ind = np.argmax(y)
    if (y[ind] < threshold):
        return
    else:
        center.append(x[ind])
        amplitude.append(y[ind])
        if (ind < ws):
            find_peaks(x[ind+ws:], y[ind+ws:], ws, threshold)
        elif (len(x) - ind < ws):
            find_peaks(x[:ind-ws], y[:ind-ws], ws, threshold)
        else:
            find_peaks(x[:ind-ws], y[:ind-ws], ws, threshold)
            find_peaks(x[ind+ws:], y[ind+ws:], ws, threshold)
                 
for j in range(repeat):
    np.random.seed(j)
    #center_set = [1.5]
    center_set = np.random.random_sample(num_steps) + 1
    center_set.sort()
    y = np.random.randn(num_dps)
    for i in range(len(center_set)):
        y[(x > center_set[i])] += step
    z = np.concatenate((np.ones(50), np.ones(50) * -1), axis = 0)
    convolved = scipy.signal.convolve(y, z, mode = 'valid') / 50
    x_new = x[50: -49]
    find_peaks(x_new, convolved, 50, 1)
    print('case: ', j)
    print('center_set: ', center_set)
    print("center_found: ", center)
    print("amplitude_found: ", amplitude)
    center = []
    amplitude = []