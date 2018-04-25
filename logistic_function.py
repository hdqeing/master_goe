import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from inspect import signature
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import os

num_dps = 1000 #number of data points
ws = 100 #window size

x = np.linspace(0,2,num_dps)

def logistic(x, c, delta, k, t0): #logistic function
    return c + delta / (1 + np.exp(-k * (x - t0)))

def sliding_window(x, y, func, initial_values, window_size):
    '''
    A sliding window method. A window with a given size slides through the curve, data in the interval are fit with a given function (defined in advance).
    Return:
        A list of parameters.
    '''
    length = len(x) - window_size + 1
    sig = signature(func)
    params = sig.parameters
    num_params = len(params)
    params_calc = np.zeros((length,num_params))
    for i in range(length):
        params_calc[i][0] = np.mean(x[i : i + window_size])
        try:
            if len(initial_values) == length:
                params_calc[i][1:] = curve_fit(func, x[i : i + window_size], y[i : i + window_size], initial_values[i])[0]
            elif len(initial_values) == 1:
                params_calc[i][1:] = curve_fit(func, x[i : i + window_size], y[i : i + window_size], initial_values)[0]
            else:
                print("Invalid initial values!")
                return
        except RuntimeError:
            params_calc[i][1:] = np.nan
    return params_calc

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

def plot_combi(x_org, y_org, x, y, z, x_label, y_label, z_label, fig_title = "whatever"):
    fig = plt.figure()
    gs = gridspec.GridSpec(5, 4)

    ax1 = fig.add_subplot(gs[0,1:3])
    ax1.plot(x_org, y_org)
    ax1.set_title(fig_title)
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2 = fig.add_subplot(gs[1:3,1:3])
    splt2 = ax2.plot(x, y, "og", label = y_label)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label)
    ax3 = ax2.twinx()
    splt3 = ax3.plot(x, z, "ob", label = z_label)
    ax3.set_ylabel(z_label)
    splt = splt2 + splt3
    lbls = [l.get_label() for l in splt]
    ax2.legend(splt, lbls, loc = 0)

    ax4 = fig.add_subplot(gs[1:3, 0])
    ax4.hist(y, orientation = "horizontal")
    x_lim = ax4.get_xlim()
    ax4.set_xlim((x_lim[1], x_lim[0]))
    plt.setp(ax4.get_yticklabels(), visible=False)

    ax5 = fig.add_subplot(gs[1:3, 3])
    ax5.hist(z, orientation = "horizontal")
    plt.setp(ax5.get_yticklabels(), visible=False)

    return fig, gs, [ax1, ax2, ax3, ax4, ax5] 

#varying amplitude, see how robust the method is.

amplitude_set_1 = np.arange(0, 1, 0.1)
amplitude_set_2 = np.arange(1, 2, 0.2)
amplitude_set_3 = np.arange(2, 3, 0.3)
amplitude_set_4 = np.arange(3, 4, 0.4)
amplitude_set_5 = np.arange(4, 5.1, 0.5)
amplitude_set = np.concatenate((amplitude_set_1, amplitude_set_2, amplitude_set_3, amplitude_set_4, amplitude_set_5), axis = 0)

amp_calc = np.zeros(len(amplitude_set))
cen_calc = np.zeros(len(amplitude_set))

amp_found = np.zeros((len(amplitude_set), 100))
cen_found = np.zeros((len(amplitude_set), 100))

for i in range(len(amplitude_set)):
    for j in range(100):
        bins = 20
        filename = "/home/qing/projects/step_detection/logistic_function/data/amp_"+str(amplitude_set[i])+"_case_"+str(j)+".npy"
        ps_eff, cen_eff, amp_eff = np.load(filename)
        cen = moving_median(cen_eff, 49)
        amp = moving_median(amp_eff, 49)
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
        fig, gs, axes = plot_combi([0,1], [0,1], ps_eff, cen_eff, amp_eff, "position", "center", "amplitude", fig_title = "amplitude = " + str(amplitude_set[i]))
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
    plt.plot(cen_found[i], amp_found[i], '.', color = "green")
    plt.scatter(1, amplitude_set[i],color = "red")
    plt.title("amplitude_set: " + str(amplitude_set[i]))
    plt.xlim((0,2))
    plt.show()




sys.exit(0)

ps = moving_median(ps_eff, 50)
cen = moving_median(cen_eff, 50)
amp = moving_median(amp_eff, 50)

for j in range(len(amplitude_set)):
    for k in range(100):
        y = np.random.standard_normal(num_dps)
        y[x > 1] += amplitude_set[j]
        #prepare initial value array
        initial_values = np.zeros((len(x) - ws + 1, 4))
        for i in range(len(initial_values)):
            initial_values[i][0] = np.mean(y[i : i + ws])
            initial_values[i][1] = max(y[i : i + ws]) - min(y[i : i + ws])
            initial_values[i][2] = 50
            initial_values[i][3] = np.mean(x[i : i + ws])
        fit_result = sliding_window(x, y, logistic, initial_values, ws)
        results_filtered = filter_logistic_fit_result(x, y, fit_result, ws)
        position = results_filtered[:,0]
        amplitude = results_filtered[:,2]
        center = results_filtered[:,-1]
        ps_eff = position[~np.isnan(amplitude)]
        cen_eff = center[~np.isnan(amplitude)]
        amp_eff = amplitude[~np.isnan(amplitude)]
        np.save("amp_" + str(amplitude_set[j]) + "_case_" + str(k), np.array([ps_eff, cen_eff, amp_eff]))