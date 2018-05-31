import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from inspect import signature
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

def plot_combi(x_org, y_org, x, y, z, bins, x_label, y_label, z_label, fig_title = "whatever"):
    fig = plt.figure()
    gs = gridspec.GridSpec(5, 4)

    #original data
    ax1 = fig.add_subplot(gs[0,1:3])
    ax1.plot(x_org, y_org)
    ax1.set_title(fig_title, fontsize = 20)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.tick_params(axis = 'both', labelsize = 16)

    #parameter 1 against window position
    ax2 = fig.add_subplot(gs[1:3,1:3], sharex = ax1)
    splt2 = ax2.plot(x, y, "og", label = y_label)
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.set_ylabel(y_label, fontsize = 20)
    ax2.tick_params(axis = 'both', labelsize = 16)

    #parameter 2 against window position
    ax3 = ax2.twinx()
    splt3 = ax3.plot(x, z, "ob", label = z_label)
    ax3.set_ylabel(z_label, fontsize = 20)
    ax3.tick_params(axis = 'both', labelsize = 16)
    splt = splt2 + splt3
    lbls = [l.get_label() for l in splt]
    ax2.legend(splt, lbls, loc = 0, prop = {'size':20})

    #histogram of parameter 1
    ax4 = fig.add_subplot(gs[1:3, 0])
    ax4.hist(y,bins = bins, orientation = "horizontal")
    x_lim = ax4.get_xlim()
    ax4.set_xlim((x_lim[1], x_lim[0]))
    ax4.tick_params(axis = 'both', labelsize = 16)
    plt.setp(ax4.get_yticklabels(), visible=False)

    #histogram of parameter 2
    ax5 = fig.add_subplot(gs[1:3, 3])
    ax5.hist(z, bins = bins, orientation = "horizontal")
    ax5.tick_params(axis = 'both', labelsize = 16)
    plt.setp(ax5.get_yticklabels(), visible=False)

    #histogram 2D
    ax6 = fig.add_subplot(gs[3:5, 1:3], sharex = ax1)
    H, cen_edge, amp_edge = np.histogram2d(y, z, bins)
    H = H.T
    X, Y = np.meshgrid(cen_edge, amp_edge)
    ax6.set_xlabel("center calculated", fontsize = 20)
    ax6.set_ylabel("amplitude calculated", fontsize = 20)
    histo = ax6.pcolormesh(X, Y, H)
    ax6.plot(y, z, "og")
    ax6.tick_params(axis = 'both', labelsize = 20)

    #colorbar
    ax7 = fig.add_subplot(gs[3:5, 3])
    divider = make_axes_locatable(ax7)
    cax = divider.append_axes("left", size = "5%", pad = 0.08)
    cbar = fig.colorbar(histo, cax = cax)
    cbar.ax.tick_params(labelsize = 20)
    ax7.axis("off")
    
    fig.subplots_adjust(left=0.01, bottom=0.1, right=0.99, top=0.95,wspace=0.3, hspace=0.1)

    return fig

def find_peaks(x, y, bins, tolerance, ws):
    '''
    This function search for subgroups with population larger than ws*tolerance.

    Assumption:
        1. the window size is chosen carefully so that there is only one step in the window.
        2. "curve_fit" function is reliable, which means whenever the real step is located inside the window, curve_fit function return the amplitude and step of the real step.
        3. a tolerance is given considering that there might be too few data points on the left (right) side of the step, when the window start to include (or is about to exclude) the step

    Arguements:
        x: array, calculated center.
        y: array, calculated amplitude.
        bins: int, bins for making 2D histogram.
        tolerance: float, [0, 1).
        ws: int, window size for sliding window 

    '''
    H, x_edge, _ = np.histogram2d(x, y, bins)
    num_dps_same_center = np.zeros(len(H))
    cen_trial = np.array([])
    amp_trial = np.array([])
    for i in range(len(num_dps_same_center)):
        num_dps_same_center[i] = np.sum(H[i]) #same x, all y, because y is not reliable
        if (num_dps_same_center[i] > (1 - tolerance) * ws):
            cen_trial = np.append(cen_trial, np.mean(x[(x > x_edge[i]) & (x <x_edge[i + 1])]))
            amp_trial = np.append(amp_trial, np.mean(y[(x > x_edge[i]) & (x <x_edge[i + 1])]))
    return cen_trial, amp_trial