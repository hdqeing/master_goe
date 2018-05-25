from logistic_function import *
import sys

num_dps = 1000

center_candidate = []

x = np.linspace(1,2,1000)

num_steps = 2 #number of steps

step = 3

repeat = 1

def find_multiple_steps(x, y, Min_dps):
    '''
    This function generates possible center with a recursive method.
    '''
    # if there are not enough data points to fit (at least 4 data points), end 
    if (len(x) < Min_dps):
        return
    #try fit with logistic function, if an error were returned, set all parameters to 0
    try:
        c, delta, k, t0 = curve_fit(logistic,x,y,[np.mean(y), max(y) - min(y), 500, np.mean(x)])[0]
    except RuntimeError:
        c, delta, k, t0 = 0, 0, 0, 0
    #several constraints to tell if the found step is a real step or only noise profile
    if ((abs(delta) < 0.8) | (delta > max(y) - min(y)) | (t0 > max(x)) | (t0 < min(x))):
        find_multiple_steps(x[len(x)//2:], y[len(x)//2:], Min_dps)
        find_multiple_steps(x[:len(x)//2], y[:len(x)//2], Min_dps)
    else:
        center_candidate.append(t0)
        find_multiple_steps(x[x >= t0], y[x >= t0], Min_dps)
        find_multiple_steps(x[x <= t0], y[x <= t0], Min_dps)

def calculate_params(x, y, x_trial):
    '''
    This function calculate the parameter of a step more accurately.
    '''
    x_trial = np.array(x_trial)
    x_trial.sort()
    result = np.zeros((len(x_trial), 4))
    # we have 1000 data points, the span on x axis is (x[-1] - x[0]), so the element unit for x axis should be 0.01. 
    # say we find two steps with a distance smaller than 0.01, it's not reliable
    dist = np.diff(x_trial)
    print(dist)
    ind = np.where(dist < 10 * (x[-1] - x[0])/num_dps)[0]
    if len(ind) != 0:
        for i in range(len(ind)):
            x_trial[ind[i]] = np.mean([x_trial[ind[i]], x_trial[ind[i]+1]])
        for i in range(len(ind)):
            np.delete(x_trial, ind[i] + 1)
    vertex = np.hstack([x[0], x_trial, x[-1]])
    x_trial = np.zeros(len(vertex) - 1)
    
    for i in range(len(x_trial)):
        x_trial[i] = np.mean([vertex[i], vertex[i + 1]])
    print('guess: ', vertex)
    print('interval: ', x_trial)    
    #divide the curve at candidate steps, fit again, to get more accurate parameters
    for i in range(len(x_trial) - 1):
        x_interval = x[(x > x_trial[i]) & (x < x_trial[i+1])]
        y_interval = y[(x > x_trial[i]) & (x < x_trial[i+1])]
        initial_value = [np.mean(y_interval), max(y_interval) - min(y_interval), 500, np.mean(x_interval)]
        try:
            result[i] = curve_fit(logistic, x_interval, y_interval,initial_value)[0]
        except RuntimeError:
            result[i] = 0, 0, 0, 0
        if ((abs(result[i][1]) < 0.8) | (result[i][1] > max(y_interval) - min(y_interval)) | (result[i][3] > max(x_interval)) | (result[i][3] < min(x_interval))):
            result[i] = 0, 0, 0, 0
    return result

result_overall = []

for j in range(repeat):
    #generate a curve with steps
    center = np.random.random_sample(num_steps) + 1
    center.sort()
    y = np.random.randn(num_dps)
    for i in range(len(center)):
        y[(x > center[i])] += step
    find_multiple_steps(x, y, 100)
    result = calculate_params(x, y, center_candidate)
    print(center)
    print(result[:, -1])
    ind = np.where(result[:, 0] == 0)[0]
    '''
    center_candidate = np.array(center_candidate)
    center_candidate.sort()
    center_candidate = np.hstack([x[0], center_candidate, x[-1]])
    suspecious_interval = []
    for i in range(len(ind)):
        suspecious_interval.append([center_candidate[ind[i]], center_candidate[ind[i] + 2]])
    
    for i in range(len(suspecious_interval)):
        x_interval = x[(x > suspecious_interval[i][0]) & (x < suspecious_interval[i][1])]
        y_interval = y[(x > suspecious_interval[i][0]) & (x < suspecious_interval[i][1])]
        center_candidate = []
        find_multiple_steps(x_interval, y_interval, 20)
        print(suspecious_interval[i])
        print(center_candidate)
        result_new = calculate_params(x_interval, y_interval, center_candidate)
        print(result_new)
    #plot results
    plt.plot(x, y)
    center_predicted = np.hstack([x[0], result[:, 3], x[-1]])
    center_predicted.sort()
    for i in range(len(center_predicted) - 2):
        x_interval = x[(x > center_predicted[i]) & (x < center_predicted[i+2])]
        plt.plot(x_interval, logistic(x_interval, result[i][0], result[i][1], result[i][2], result[i][3]))
    plt.show()
    '''
    result_overall.append(result)
    center_candidate = []