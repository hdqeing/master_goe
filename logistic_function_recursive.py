from logistic_function import *
import sys

num_dps = 1000

center_candidate = []

x = np.linspace(1,2,1000)

num_steps = 10 #number of steps

step = 3

repeat = 1

def find_multiple_steps(x, y, Min_dps):
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

result_overall = []

#10 steps
for j in range(repeat):
    #generate a curve with steps
    center = np.random.random_sample(num_steps) + 1
    center.sort()
    print(center)
    y = np.random.randn(1000)
    for i in range(len(center)):
        y[(x > center[i])] += step
    find_multiple_steps(x, y, 50)
    #we have 1000 data points, the span on x axis is 1, so the unit for x axis should be 0.01. 
    # say we find two steps with a distance smaller than 0.01, it's not reliable
    center_candidate = np.array(center_candidate)
    center_candidate.sort()
    dist = np.diff(center_candidate)
    ind = np.where(dist < (x[-1] - x[0])/num_dps)[0]
    if len(ind) != 0:
        for i in range(len(ind)):
            center_candidate[ind[i]] = np.mean([center_candidate[ind[i]], center_candidate[ind[i]+1]])
        for i in range(len(ind)):
            np.delete(center_candidate, ind[i] + 1)
    print(center_candidate)
    result = np.zeros((len(center_candidate), 4))
    center_candidate = np.hstack([x[0], center_candidate, x[-1]])
    #divide the curve at candidate steps, fit again, to get more accurate parameters
    for i in range(len(center_candidate) - 2):
        x_interval = x[(x > center_candidate[i]) & (x < center_candidate[i+2])]
        y_interval = y[(x > center_candidate[i]) & (x < center_candidate[i+2])]
        initial_value = [np.mean(y_interval), max(y_interval) - min(y_interval), 500, np.mean(x_interval)]
        try:
            result[i] = curve_fit(logistic, x_interval, y_interval,initial_value)[0]
        except RuntimeError:
            result[i] = 0, 0, 0, 0
        if ((abs(result[i][1]) < 0.8) | (result[i][1] > max(y_interval) - min(y_interval)) | (result[i][3] > max(x_interval)) | (result[i][3] < min(x_interval))):
            result[i] = 0, 0, 0, 0
    ind_ineff = np.where(result[:,0] == 0)
    print(ind_ineff)
    result = np.delete(result, ind_ineff, 0)
    print(result)
    plt.plot(x, y)
    center_predicted = np.hstack([x[0], result[:, 3], x[-1]])
    center_predicted.sort()
    for i in range(len(center_predicted) - 2):
        x_interval = x[(x > center_predicted[i]) & (x < center_predicted[i+2])]
        plt.plot(x_interval, logistic(x_interval, result[i][0], result[i][1], result[i][2], result[i][3]))
    plt.show()
        

    result_overall.append(result)
    center_candidate = []


sys.exit()

center_1 = 1.2
center_2 = 1.5

step_1 = 3
step_2 = 6

if (center_candidate[0] == max(center_candidate)):
    center_candidate.remove(center_candidate[0])
    find_multiple_steps(x[x > max(center_candidate)], y[x > max(center_candidate)], 100)
if (center_candidate[0] == min(center_candidate)):
    center_candidate.remove(center_candidate[0])
    find_multiple_steps(x[x < min(center_candidate)], y[x < min(center_candidate)], 100)
B = center_candidate[1]
center_candidate_array = np.array(center_candidate)
A = center_candidate_array[center_candidate_array < center_candidate_array[0]][0]
if ((center_candidate[0] > A) & (center_candidate[0] < B)):
    center_candidate.remove(center_candidate[0])
    find_multiple_steps(x[(x > A) & (x < B)], y[(x > A) & (x < B)], 100)

print(center_candidate)
print(center)











if ((len(center_candidate) == 3) & (center_candidate[0] < center_candidate[1]) & (center_candidate[0] > center_candidate[2])):
    print('first step not reliable')
    center_candidate.remove(center_candidate[0])
    find_multiple_steps(x[(x > center_candidate[2]) & (x < center_candidate[1])], y[(x > center_candidate[2]) & (x < center_candidate[1])])
elif ((len(center_candidate) == 2) & (center_candidate[1] > center_candidate[0])):
    print('this is a refit')
    center_candidate.remove(center_candidate[0])
    find_multiple_steps(x[x > center_candidate[0]], y[x > center_candidate[0]])
elif ((len(center_candidate) == 2) & (center_candidate[1] < center_candidate[0])):
    print('this is a refit')
    center_candidate.remove(center_candidate[0])
    find_multiple_steps(x[x < center_candidate[0]], y[x < center_candidate[0]])




#record the result, divide the curve at the step found in former calculation(t0), fit the two part individually 
for i in range(20):
    y = np.random.randn(1000)
    y[(x > center_1) & (x < center_2)] += step_1
    y[x > center_2] += step_2
    find_multiple_steps(x,y,i)

for i in range(len(step_found)):
    for j in range(len(step_found[i])):
        plt.plot(x, logistic(x, const_found[i][j], step_found[i][j], k_found[i][j], center_candidate[i][j]))
    print(center_candidate[i])
    print(k_found[i])
    plt.plot(x, y)
    plt.show()

'''
for i in range(20):
    plt.plot([center_1, center_2], [step_1, step_2 - step_1], 'or', label = "set")
    plt.plot(center_candidate[i], step_found[i], 'og', label = "calculated")
    plt.title("case: "+ str(i))
    plt.legend()
    plt.show()
'''