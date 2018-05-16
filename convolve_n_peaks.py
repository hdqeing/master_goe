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
    center = np.array([])
    step = np.array([])
    y_messed_up = np.copy(y)
    ind = np.argmax(y_messed_up)
    while (y[ind] > threshold):
        center = np.append(center, x[ind])
        step = np.append(step, y[ind])
        #print('center:', x[ind] , 'step:', y[ind], '\n')
        if (ind - ws > 0):
            y_messed_up[ind - ws : ind] = 0
        else:
            y_messed_up[: ind] = 0
        if (ind + ws < len(y)):
            y_messed_up[ind : ind + ws] = 0
        else:
            y_messed_up[ind : ] = 0
        ind = ind = np.argmax(y_messed_up)

    plt.plot(x, y)
    for j in range(len(center)):
        plt.scatter(center[j], step[j], color = 'red')
    plt.show()
    return center, step

for i in range(20):
    y = np.random.randn(1000)
    y[(x > center_1) & (x < center_2)] += step_1
    y[x > center_2] += step_2
    z = np.concatenate((np.ones(50), np.ones(50) * -1), axis = 0)
    convolved = scipy.signal.convolve(y, z, mode = 'valid') / 50
    x_new = x[50: -49]
    print(i)
    center, step = find_peaks(x_new, convolved, 50, 0.5)