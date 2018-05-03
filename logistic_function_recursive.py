from logistic_function import *

step_found = [[] for i in range(20)]
center_found = [[] for i in range(20)]

x = np.linspace(1,2,1000)

center_1 = 1.2
center_2 = 1.5

step_1 = 3
step_2 = 6

def find_multiple_steps(x, y, i):
    # if there are not enough data points to fit (at least 4 data points), end 
    if (len(x) < 4):
        return
    #try fit with logistic function, if an error were returned, set all parameters to 0
    try:
        c, delta, k, t0 = curve_fit(logistic,x,y,[np.mean(y), max(y) - min(y), 500, np.mean(x)])[0]
    except RuntimeError:
        c, delta, k, t0 = 0, 0, 0, 0
    #several constraints to tell if the found step is a real step or only noise profile
    if ((delta < 0.8) | (delta > max(y)) | (t0 > max(x)) | (t0 < min(x))):
        return
    '''
    ???
    The constraints do not necessarily imply that there is no steps in the curve.
    It is possible that there are still one or more steps in the curve, but the logistic function cannot detect. 
    '''
    #record the result, divide the curve at the step found in former calculation(t0), fit the two part individually 
    else:
        step_found[i].append(delta)
        center_found[i].append(t0)
        find_multiple_steps(x[x > t0], y[x > t0], i)
        find_multiple_steps(x[x < t0], y[x < t0], i)

for i in range(20):
    y = np.random.randn(1000)
    y[(x > center_1) & (x < center_2)] += step_1
    y[x > center_2] += step_2
    find_multiple_steps(x,y,i)

for i in range(20):
    plt.plot([center_1, center_2], [step_1, step_2 - step_1], 'or', label = "set")
    plt.plot(center_found[i], step_found[i], 'og', label = "calculated")
    plt.title("case: "+ str(i))
    plt.legend()
    plt.show()