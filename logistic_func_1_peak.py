from logistic_function import *
from matplotlib.patches import Rectangle
import sys

num_dps = 1000 #number of data points
ws = 100 #window size

x = np.linspace(1,2,num_dps)

step = np.around(np.exp(np.linspace(0, 2, 25)) - 1, decimals = 1)

amp_calc = np.zeros(len(step))
cen_calc = np.zeros(len(step))

amp_found = np.zeros((len(step), 100))
cen_found = np.zeros((len(step), 100))

def on_pick(event):
    index = int(event.ind)
    #filename = "/home/qing/projects/step_detection/logistic_function/data/amp_"+str(step[i])+"_case_"+str(index)+".npy"
    filename = "/home/whatever/PythonProjects/DwellAnalysis/logistic_function/data/amp_"+str(step[i])+"_case_"+str(index)+".npy"
    ps_eff, cen_eff, amp_eff = np.load(filename)
    cen = moving_median(cen_eff, 49)
    amp = moving_median(amp_eff, 49)
    bins = 20
    H, cen_edge, amp_edge = np.histogram2d(cen, amp, bins)
    ind = np.where(H == np.amax(H))
    H = H.T
    #2D histogram of cen and amp
    while(len(ind[0]) > 1):
        bins += 1
        H, cen_edge, amp_edge = np.histogram2d(cen, amp, bins)
        ind = np.where(H == np.amax(H))
        H = H.T
    figi, gs, axes = plot_combi([0,1], [0,1], ps_eff, cen_eff, amp_eff,20, "position", "center", "amplitude", fig_title = "amplitude = " + str(step[i]))
    ax = figi.add_subplot(gs[3:5, 1:3])
    X, Y = np.meshgrid(cen_edge, amp_edge)
    ax.set_xlabel("center calculated")
    ax.set_ylabel("amplitude calculated")
    histo = ax.pcolormesh(X, Y, H)
    ax1 = figi.add_subplot(gs[3:5, 3])
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("left", size = "5%", pad = 0.08)
    figi.colorbar(histo, cax = cax)
    ax1.axis("off")
    ax.plot(cen, amp, "og")
    figi.show()

for i in range(len(step)):
    for j in range(100):
        bins = 20
        #pc location
        #filename = "/home/qing/projects/step_detection/logistic_function/data/amp_"+str(step[i])+"_case_"+str(j)+".npy"
        #lab computer location
        filename = "/home/whatever/PythonProjects/DwellAnalysis/logistic_function/data/amp_"+str(step[i])+"_case_"+str(j)+".npy"
        ps_eff, cen_eff, amp_eff = np.load(filename)
        cen = moving_median(cen_eff, 49)
        amp = moving_median(amp_eff, 49)
        #single peak
        H, cen_edge, amp_edge = np.histogram2d(cen, amp, bins)
        ind = np.where(H == np.amax(H))
        H = H.T
        #2D histogram of cen and amp
        while(len(ind[0]) > 1):
            bins += 1
            H, cen_edge, amp_edge = np.histogram2d(cen, amp, bins)
            ind = np.where(H == np.amax(H))
            H = H.T
        cen_found[i][j] = np.nanmean(cen[(cen >= cen_edge[int(ind[0])]) & (cen <= cen_edge[int(ind[0] + 1)])])
        amp_found[i][j] = np.nanmean(amp[(amp >= amp_edge[int(ind[1])]) & (amp <= amp_edge[int(ind[1] + 1)])])
        #plot
        '''
        fig, gs, axes = plot_combi([0,1], [0,1], ps_eff, cen_eff, amp_eff, "position", "center", "amplitude", fig_title = "amplitude = " + str(step[i]))
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
        plt.show()
        '''
    fig, ax1 = plt.subplots()
    ax2 = fig.add_axes(ax1)
    ax1.plot(cen_found[i], amp_found[i], '.', label = 'found step for individual test case', picker = 5)
    ax1.plot([1.5], [step[i]], 'or', label = 'set step')
    ax1.set_xlabel('found center', fontsize = 20)
    ax1.set_ylabel('found amplitude', fontsize = 20)
    ax1.set_xlim([1,2])
    p = Rectangle((1.5 / 1.05, step[i] / 1.05), width = 1.5 / 0.95 - 1.5 / 1.05, height = step[i] / 0.95 - step[i] / 1.05, alpha = 0.5, edgecolor = 'red', facecolor = 'red', label = 'error < 5%')
    q = Rectangle((1.5 / 1.01, step[i] / 1.01), width = 1.5 / 0.99 - 1.5 / 1.01, height = step[i] / 0.99 - step[i] / 1.01, alpha = 0.5, edgecolor = 'green', facecolor = 'green', label = "error < 1%")
    ax2.add_patch(p)
    ax2.add_patch(q)
    ax1.set_title('Distribution of found step for each test case at set amplitude: ' + str(step[i]), fontsize = 20)
    ax1.tick_params(axis = 'both', labelsize = 20)
    ax1.legend(prop = {'size':20})
    figManager = plt.get_current_fig_manager()
    figManager.resize(*figManager.window.maxsize())
    cid = fig.canvas.mpl_connect('pick_event', on_pick)
    plt.show()

sys.exit()

#how many of the test cases have an error smaller than 1% or 5%
cen_error_1 = np.zeros(len(step))
amp_error_1 = np.zeros(len(step))
cen_error_5 = np.zeros(len(step))
amp_error_5 = np.zeros(len(step))

for i in range(len(step)):
    for j in range(100):
        bins = 20
        #pc location
        #filename = "/home/qing/projects/step_detection/logistic_function/data/amp_"+str(step[i])+"_case_"+str(j)+".npy"
        #lab computer location
        filename = "/home/whatever/PythonProjects/DwellAnalysis/logistic_function/data/amp_"+str(step[i])+"_case_"+str(j)+".npy"
        ps_eff, cen_eff, amp_eff = np.load(filename)
        cen = moving_median(cen_eff, 49)
        amp = moving_median(amp_eff, 49)
        #single peak
        H, cen_edge, amp_edge = np.histogram2d(cen, amp, bins)
        ind = np.where(H == np.amax(H))
        H = H.T
        #2D histogram of cen and amp
        while(len(ind[0]) > 1):
            bins += 1
            H, cen_edge, amp_edge = np.histogram2d(cen, amp, bins)
            ind = np.where(H == np.amax(H))
            H = H.T
        cen_found[i][j] = np.nanmean(cen[(cen >= cen_edge[int(ind[0])]) & (cen <= cen_edge[int(ind[0] + 1)])])
        amp_found[i][j] = np.nanmean(amp[(amp >= amp_edge[int(ind[1])]) & (amp <= amp_edge[int(ind[1] + 1)])])
    cen_found[i] = cen_found[i][~np.isnan(cen_found[i])]
    amp_found[i] = amp_found[i][~np.isnan(amp_found[i])]
    error_center = abs(cen_found[i] - 1.5) / 1.5
    error_step = abs(amp_found[i] - step[i]) / step[i]
    cen_error_1[i] = len(error_center[error_center < 0.01])/ len(error_center)
    amp_error_1[i] = len(error_step[error_step < 0.01])/ len(error_step)
    cen_error_5[i] = len(error_center[error_center < 0.05])/ len(error_center)
    amp_error_5[i] = len(error_step[error_step < 0.05])/ len(error_step)
    print('center error: ', step[i], cen_error_1[i], cen_error_5[i])
    print('amplitude error: ', step[i], amp_error_1[i], amp_error_5[i])

plt.plot(step, cen_error_1, label = "error < 1 %")
plt.plot(step, cen_error_5, label = "error < 5 %")
plt.title("percentage of test cases, whose predicted center has an error smaller than a given value", fontsize = 20)
plt.legend(prop = {'size' : 20})
plt.tick_params(axis = 'both', labelsize = 20)
plt.xlabel("set amplitude", fontsize = 20)
plt.ylabel('percentage', fontsize = 20)
plt.show()
plt.plot(step, amp_error_1, label = "error < 1 %")
plt.plot(step, amp_error_5, label = "error < 5 %")
plt.title("percentage of test cases, whose predicted amplitude has an error smaller than a given value", fontsize = 20)
plt.legend(prop = {'size' : 20})
plt.tick_params(axis = 'both', labelsize = 20)
plt.xlabel("set amplitude", fontsize = 20)
plt.ylabel('percentage', fontsize = 20)
plt.show()

sys.exit()

ps = moving_median(ps_eff, 50)
cen = moving_median(cen_eff, 50)
amp = moving_median(amp_eff, 50)

#sliding window, fit with logistic function, save data 
for j in range(len(step)):
    for k in range(100):
        y = np.random.standard_normal(num_dps)
        y[x > 1] += step[j]
        #prepare initial value array
        initial_values = np.zeros((len(x) - ws + 1, 4))
        for i in range(len(initial_values)):
            initial_values[i][0] = np.mean(y[i : i + ws])
            initial_values[i][1] = max(y[i : i + ws]) - min(y[i : i + ws])
            initial_values[i][2] = 50
            initial_values[i][3] = np.mean(x[i : i + ws])
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
        np.save("amp_" + str(step[j]) + "_case_" + str(k), np.array([ps_eff, cen_eff, amp_eff]))

#find multiple steps (even though the data set only contains one single step)
for i in range(len(step)):
    for j in range(100):
        filename = "/home/whatever/PythonProjects/DwellAnalysis/logistic_function/data/amp_"+str(step[i])+"_case_"+str(j)+".npy"
        ps_eff, cen_eff, amp_eff = np.load(filename)
        cen = moving_median(cen_eff, 49)
        amp = moving_median(amp_eff, 49)
        cen_found, amp_found = find_peaks(cen, amp, 15, 80)
        if (len(cen_found) > 1):
            print("set step at 1.5 with amplitude " + str(step[i]) + " test case " + str(j))
            for k in range(len(cen_found)):
                print(str(k+1) + ". step found at " + str(cen_found[k])  + " with amplitude " + str(amp_found[k]))