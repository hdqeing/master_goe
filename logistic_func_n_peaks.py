from logistic_function import *
import sys

ws = 100

num_dps = 1000

x = np.linspace(1,2,1000)

center_candidate = []

x = np.linspace(1,2,1000)

num_steps = 2 #number of steps

step = 3

repeat = 10

center_1 = np.arange(1.1, 2, 0.1)
center_1 = np.around(center_1, decimals = 1)
center_2 = 1.5

x = np.linspace(1, 2, 1000)

for l in range(len(center_1)):
    for j in range(repeat):
        center_set = np.array([center_1[l], center_2])
        center_set.sort()
        y = np.random.randn(num_dps)
        for k in range(len(center_set)):
            y[(x > center_set[k])] += step
        bins = 20
        #pc location
        #filename = "/home/qing/projects/step_detection/logistic_function/data/amp_"+str(step[i])+"_case_"+str(j)+".npy"
        #lab computer location
        filename = "/home/whatever/PythonProjects/DwellAnalysis/abschluss/data/sliding_window_two_steps/location_cen1_" + str(center_1[l]) + "_case_"+ str(j) + ".npy"
        ps_eff, cen_eff, amp_eff = np.load(filename)
        ps = moving_median(ps_eff, 49)
        cen = moving_median(cen_eff, 49)
        amp = moving_median(amp_eff, 49)
        #H, cen_edge, amp_edge = np.histogram2d(cen_eff, amp_eff, bins)
        H, cen_edge, amp_edge = np.histogram2d(cen, amp, bins)
        H = H.T
        fig, gs, axes = plot_combi(x, y, ps, cen, amp, bins, "position", "center", "amplitude", fig_title = 'center_1: '+ str(center_1[l]) +'; center_2: ' + str(1.5))
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
        center, amplitude = find_peaks(cen, amp, 20, 0.5, ws)
        print(center, amplitude)
        plt.show()

sys.exit(0)

#varying the distance between 2 steps, sliding window, fit with logistic function
for l in range(len(center_1)):
    for j in range(repeat):
        center_set = np.array([center_1[l], center_2])
        center_set.sort()
        y = np.random.randn(num_dps)
        for k in range(len(center_set)):
            y[(x > center_set[k])] += step
        filename = "/home/whatever/PythonProjects/DwellAnalysis/abschluss/data/sliding_window_two_steps/location_cen1_" + str(center_1[l]) + "_case_"+ str(j) + ".npy"
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





















