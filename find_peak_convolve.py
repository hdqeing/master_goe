import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import sys
sys.path.append("/home/qing/projects/lala/DataAnalysisTools/DataAnalysis/tools")
from statistics_lib import moving_average

x = np.linspace(1,2,1000)
step = np.around(np.exp(np.linspace(0, 2, 25)) - 1, decimals = 1)
repeat = 100
ws_mv_avg = 49
ws_sq_wv = 100
a = np.ones(ws_sq_wv) / ws_sq_wv
x_convolved = scipy.signal.convolve(x, a, mode = "valid")

cen_calc = np.zeros((len(step), repeat))
amp_calc = np.zeros((len(step), repeat))

def find_peak(x, y, ws_mv_avg, threshold, reliability = False):
    """
    This function return the location and amplitude of steps, whose calculated amplitude is larger than threshold.
    The convolved result is firstly smoothed with moving average filter. 
    The points, at which the first differential change sign, are marked as the location of trial steps. 
    Were the amplitude of the trial steps greater than threshold, it would be considered as a step.

    Args:
        x: array_like.
        y: array_like.
        ws_mv_avg: window size of running average filter, can only be odd number.
        threshold: cut-off amplitude.
        reliability: boolean. If true, the reliability of every predicted result (based on amplitude) will be return.

    Returns:
        Location of steps.
        Amplitude of steps.
        If reliability is true, an array of reliability is returned. 

    """
    x_smooth = moving_average(x, ws_mv_avg, check_equidist= False)
    y_smooth = moving_average(y, ws_mv_avg, check_equidist= False)
    x_diff = np.diff(x_smooth)
    y_diff = np.diff(y_smooth)
    diff = y_diff / x_diff
    product = np.array([diff[i] * diff[i+1] for i in range(len(diff) - 1)])
    ind = np.where(product < 0)[0] + ws_mv_avg // 2
    cen_found = np.array()
    amp_found = np.array()
    relia = np.array()
    for k in range(len(ind)):
        if y[ind[k]] > threshold:
            np.append(cen_found, x[ind[i]])
            np.append(amp_found, y[ind[i]])
    if reliability:
        return cen_found, amp_found, relia
    else:
        return cen_found, amp_found


def on_pick(event):
    ind = int(event.ind)
    filename = "/home/qing/projects/step_detection/convolve/data/convolve_amp" + str(step[i]) + "_case_" + str(ind) + ".npy"
    y_convolved = np.load(filename) / (ws_sq_wv / 2)
    x_smooth = moving_average(x_convolved, ws_mv_avg, check_equidist= False)
    y_smooth = moving_average(y_convolved, ws_mv_avg, check_equidist= False)
    x_diff = np.diff(x_smooth)
    y_diff = np.diff(y_smooth)
    diff = y_diff / x_diff
    product = np.array([diff[k] * diff[k+1] for k in range(len(diff) - 1)])
    index = np.where(product < 0)[0] + ws_mv_avg // 2
    figi, [ax1, ax2] = plt.subplots(2, 1, sharex=True)
    ax1.plot(x, y)
    ax2.plot(x_convolved, y_convolved)
    ax2.plot(x_convolved[y_convolved == max(y_convolved[index])], max(y_convolved[index]), 'or')
    figi.show()

np.random.seed(19931229)

for i in range(len(step)):
    for j in range(repeat):
        filename = "/home/qing/projects/step_detection/convolve/data/convolve_amp" + str(step[i]) + "_case_" + str(j) + ".npy"
        y = np.random.randn(1000)
        y[x > 1.5] += step[i]
        y_convolved = np.load(filename) / (ws_sq_wv / 2) #convolving result
        #suppose every dataset only contains a single step (prescribed)
        x_smooth = moving_average(x_convolved, ws_mv_avg, check_equidist= False)
        y_smooth = moving_average(y_convolved, ws_mv_avg, check_equidist= False)
        x_diff = np.diff(x_smooth)
        y_diff = np.diff(y_smooth)
        diff = y_diff / x_diff
        product = np.array([diff[k] * diff[k+1] for k in range(len(diff) - 1)])
        ind = np.where(product < 0)[0] + ws_mv_avg // 2
        amp_calc[i][j] = max(y_convolved[ind])
        cen_calc[i][j] = x_convolved[y_convolved == max(y_convolved[ind])]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(cen_calc[i], amp_calc[i], '.', color = "green", label = "test sample", picker = 5)
    ax.scatter(1.5, step[i], color = "red", label = "set value")
    ax.set_title("amplitude_set: " + str(step[i]), fontsize = 20)
    ax.set_xlim((1,2))
    ax.set_xlabel("center found", fontsize = 20)
    ax.set_ylim((step[i] - 1, step[i] + 1))
    ax.set_ylabel("amplitude found", fontsize = 20)
    plt.legend()
    figManager = plt.get_current_fig_manager()
    figManager.resize(*figManager.window.maxsize())
    cid = fig.canvas.mpl_connect('pick_event', on_pick)
    plt.show()