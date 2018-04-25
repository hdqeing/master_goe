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
x = scipy.signal.convolve(x, a, mode = "valid")

cen_calc = np.zeros((len(step), repeat))
amp_calc = np.zeros((len(step), repeat))

for i in range(len(step)):
    for j in range(repeat):
        filename = "/home/qing/projects/step_detection/convolve/data/convolve_amp" + str(step[i]) + "_case_" + str(j) + ".npy"
        data = np.load(filename) / (ws_sq_wv / 2)
        x_smooth = moving_average(x, ws_mv_avg, check_equidist= False)
        y_smooth = moving_average(data, ws_mv_avg, check_equidist= False)
        x_diff = np.diff(x_smooth)
        y_diff = np.diff(y_smooth)
        diff = y_diff / x_diff
        product = np.array([diff[i] * diff[i+1] for i in range(len(diff) - 1)])
        ind = np.where(product < 0)[0] + ws_mv_avg // 2
        amp_calc[i][j] = max(data[ind])
        cen_calc[i][j] = x[data == max(data[ind])]
    plt.plot(cen_calc[i], amp_calc[i], '.', color = "green")
    plt.scatter(1.5, step[i],color = "red")
    plt.title("amplitude_set: " + str(step[i]))
    plt.xlim((1,2))
    plt.ylim((step[i] + 1, step[i] - 1))
    plt.show()