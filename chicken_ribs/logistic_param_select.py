import numpy as np
import matplotlib.pyplot as plt

step = np.around(np.exp(np.linspace(0, 2, 25)) - 1, decimals = 1)

def filter_logistic_fit_result(x, y, fit_results, window_size):
    """
    Some criteria to filter out problematic fit results, which include:
        1. the center of a found step locates outside the interval;
        2. the amplitude of a found step is larger than (y_max - y_min).
    """
    for i in range(len(fit_results)):
        if (fit_results[i][-1] < x[i]) or (fit_results[i][-1] > x[i + window_size - 1]):
            fit_results[i][:] = np.nan
        if (abs(fit_results[i][2]) > max(y) - min(y)):
            fit_results[i][:] = np.nan
    return fit_results

def moving_median(L, window_size):
    N = len(L) - window_size + 1
    y_smoothed = np.zeros(N)
    for i in range(N):
        y_smoothed[i] = np.median(L[i : i + window_size])
    return y_smoothed

num_nearby = np.zeros((len(step),100))

for i in range(len(step)):
    for j in range(100):
        filename = "/home/whatever/PythonProjects/DwellAnalysis/logistic_function/data/amp_"+str(step[i])+"_case_"+str(j)+".npy"
        ps_eff, cen_eff, amp_eff = np.load(filename)
        cen = moving_median(cen_eff, 49)
        amp = moving_median(amp_eff, 49)
        bins = 20
        H, cen_edge, amp_edge = np.histogram2d(cen, amp, bins)
        ind = np.where(H == np.amax(H))
        #2D histogram of cen and amp
        while(len(ind[0]) > 1):
            bins += 1
            H, cen_edge, amp_edge = np.histogram2d(cen, amp, bins)
            ind = np.where(H == np.amax(H))
        num_nearby[i][j] = np.sum(H[ind[0]])
    plt.hist(num_nearby[i], bins = 8)
    plt.title("step = "+str(step[i]))
    plt.show()