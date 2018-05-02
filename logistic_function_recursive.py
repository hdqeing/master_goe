from logistic_function import *

x = np.linspace(1,2,1000)

center_1 = 1.2
center_2 = 1.5

step_1 = 1.5
step_2 = 2.5

def find_multiple_steps(x, y):
    try:
        c, delta, k, t0 = curve_fit(logistic,x,y,[np.mean(y), max(y) - min(y), 500, np.mean(x)])[0]
        if (k < 50):
            c, delta, k, t0 = curve_fit(logistic, x, y, [c, delta, k, t0])[0]
    except RuntimeError:
        c, delta, k, t0 = 0, 0, 0, 0
    if ((delta < 0.5) | (delta > max(y)) | (t0 > max(x))|(t0<min(x))):
        return
    else:
        print('center: ', t0, 'amplitude: ',delta)
        y_predicted = logistic(x, c, delta, k, t0)
        plt.plot(x, y)
        plt.plot(x, y_predicted)
        plt.show()
        find_multiple_steps(x[x > t0], y[x > t0])
        find_multiple_steps(x[x < t0], y[x < t0])

for i in range(100):
    print('case: ', i)
    y = np.random.randn(1000)
    y[(x > center_1) & (x < center_2)] += step_1
    y[x > center_2] += step_2
    find_multiple_steps(x,y)
