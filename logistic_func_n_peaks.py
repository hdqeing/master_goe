from logistic_function import *
import sys

ws = 100

center = np.around(np.arange(1, 2, 0.05), decimals = 2)
center_2 = 1.5

step_1 = 2
step_2 = 4

x = np.linspace(1, 2, 1000)

for i in range(len(center)):
    center_1 = center[i]
    np.random.seed(i)
    y = np.random.randn(1000)
    if center_1 < center_2:
        y[(x > center_1) & (x < center_2)] += step_1
        y[x > center_2] += step_2
    else:
        y[x > step_1] += step_1
        y[(x < center_1) & (x > center_2)] += step_2
    bins = 30
    #pc location
    #filename = "/home/qing/projects/step_detection/logistic_function/data/amp_"+str(step[i])+"_case_"+str(j)+".npy"
    #lab computer location
    filename = "/home/whatever/PythonProjects/DwellAnalysis/logistic_function/data_multiple_peaks/location_cen1_" + str(i) + ".npy"
    ps_eff, cen_eff, amp_eff = np.load(filename)
    cen = moving_median(cen_eff, 49)
    amp = moving_median(amp_eff, 49)
    #H, cen_edge, amp_edge = np.histogram2d(cen_eff, amp_eff, bins)
    H, cen_edge, amp_edge = np.histogram2d(cen, amp, bins)
    H = H.T
    fig, gs, axes = plot_combi(x, y, ps_eff, cen_eff, amp_eff, 30, "position", "center", "amplitude", fig_title = "center_1 = " + str(center_1))
    ax = fig.add_subplot(gs[3:5, 1:3])
    X, Y = np.meshgrid(cen_edge, amp_edge)
    ax.set_xlabel("center calculated")
    ax.set_ylabel("amplitude calculated")
    histo = ax.pcolormesh(X, Y, H)
    ax1 = fig.add_subplot(gs[3:5, 3])
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("left", size = "5%", pad = 0.08)
    fig.colorbar(histo, cax = cax)
    ax1.axis("off")
    ax.plot(cen, amp, "og")
    #ax.plot(cen_eff, amp_eff, "og")
    plt.show()

sys.exit(0)

#varying the distance between 2 steps, sliding window, fit with logistic function
for j in range(len(center)):
    center_1 = center[j]
    np.random.seed(j)
    y = np.random.randn(1000)
    if (center_1 < center_2):
        y[(x > center_1) & (x < center_2)] += step_1
        y[x > center_2] += step_2
    else:
        y[x > step_1] += step_1
        y[(x < center_1) & (x > center_2)] += step_2
    filename = "/home/whatever/PythonProjects/DwellAnalysis/logistic_function/data_multiple_peaks/location_cen1_" + str(j) + ".npy"
    #prepare initial value array
    initial_values = np.zeros((len(x) - ws + 1, 4))
    for i in range(len(initial_values)):
        initial_values[i][0] = np.mean(y[i : i + ws])
        initial_values[i][1] = max(y[i : i + ws]) - min(y[i : i + ws])
        initial_values[i][2] = 50
        y_diff = abs(np.diff(y[i: i + ws]))
        initial_values[i][3] = x[i : i + ws - 1][y_diff == max(y_diff)]
    fit_result = sliding_window(x, y, logistic, initial_values, ws)
    #filter out data points that do not meet criteria (see function documentation)
    results_filtered = filter_logistic_fit_result(x, y, fit_result, ws)
    position = results_filtered[:,0]
    amplitude = results_filtered[:,2]
    center = results_filtered[:,-1]
    #filter out np.nan (otherwise array not taken by np.histogram, plt.hist or so)
    ps_eff = position[~np.isnan(amplitude)]
    cen_eff = center[~np.isnan(amplitude)]
    amp_eff = amplitude[~np.isnan(amplitude)]
    np.save(filename, np.array([ps_eff, cen_eff, amp_eff]))

sys.exit(0)

















