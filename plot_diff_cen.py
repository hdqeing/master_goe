import numpy as np
import scipy.stats
import scipy.signal
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys

#influence of center location

center_1 = np.linspace(1, 1.05, 20)
center_2 = np.linspace(1.055, 1.949, 10)
center_3 = np.linspace(1.95, 2, 10)
center = np.hstack([center_1, center_2, center_3])

# calculate how many data points are there on each side of step
x = np.linspace(1, 2, 1000)
num_dps_left = np.zeros(len(center))
num_dps_right = np.zeros(len(center))

for i in range(len(center)):
    num_dps_right[i] = len(x[x > center[i]])
    num_dps_left[i] = 1000 - num_dps_right[i]

result_logistic_function = np.load("/home/whatever/PythonProjects/DwellAnalysis/abschluss/data/logistic_function_cen.npy")
result_convolve = np.load("/home/whatever/PythonProjects/DwellAnalysis/abschluss/data/result_convolution_cen.npy")

#performance of logistic function at different center
for i in range(len(result_convolve)):
    fig, ax1 = plt.subplots()
    ax2 = fig.add_axes(ax1)
    ax1.plot(result_logistic_function[i,:,3], result_logistic_function[i,:,1], '.', label = 'found step for individual test case')
    ax1.plot([center[i]], [2], 'or', label = 'set step')
    ax1.set_xlabel('found center', fontsize = 20)
    ax1.set_ylabel('found amplitude', fontsize = 20)
    ax1.set_xlim([1,2])
    p = Rectangle((center[i] / 1.05, 2 / 1.05), width = center[i] / 0.95 - center[i] / 1.05, height = 2 / 0.95 - 2 / 1.05, alpha = 0.5, edgecolor = 'red', facecolor = 'red', label = 'error < 5%')
    q = Rectangle((center[i] / 1.01, 2 / 1.01), width = center[i] / 0.99 - center[i] / 1.01, height = 2 / 0.99 - 2 / 1.01, alpha = 0.5, edgecolor = 'green', facecolor = 'green', label = "error < 1%")
    ax2.add_patch(p)
    ax2.add_patch(q)
    ax1.set_title('Distribution of found step when ' + str(int(num_dps_left[i])) + ' points lie on the left of step', fontsize = 20)
    ax1.tick_params(axis = 'both', labelsize = 20)
    ax1.legend(prop = {'size':20})
    plt.show()



sys.exit()

percent_eff_logistic_func = np.zeros(len(center))

#robustness of logistic function at different center
for i in range(len(result_logistic_function)):
    delta = result_logistic_function[i,:,1]
    percent_eff_logistic_func[i] = (delta.size - np.count_nonzero(np.isnan(delta)))/len(delta)

plt.plot(num_dps_left, percent_eff_logistic_func, '-o')
plt.title('percentage of cases, where regression parameter can be found', fontsize = 20)
plt.xlabel('number of data points on the left side of step', fontsize = 20)
plt.ylabel('percentage', fontsize = 20)
plt.tick_params(axis = 'both', labelsize = 20)
plt.show()

#how many of the test cases have an error smaller than 1% or 5%
cen_error_1 = np.zeros(len(center))
amp_error_1 = np.zeros(len(center))
cen_error_5 = np.zeros(len(center))
amp_error_5 = np.zeros(len(center))

for i in range(len(result_convolve)):
    center_calc = result_logistic_function[i,:,3]
    center_calc = center_calc[~np.isnan(center_calc)]
    step_calc = result_logistic_function[i,:,1]
    step_calc = step_calc[~np.isnan(step_calc)]
    error_center = abs(center_calc - center[i]) / center[i]
    error_step = abs(step_calc - 2) / 2
    cen_error_1[i - 1] = len(error_center[error_center < 0.01])/ len(error_center)
    amp_error_1[i - 1] = len(error_step[error_step < 0.01])/ len(error_step)
    cen_error_5[i - 1] = len(error_center[error_center < 0.05])/ len(error_center)
    amp_error_5[i - 1] = len(error_step[error_step < 0.05])/ len(error_step)

plt.plot(num_dps_left, cen_error_1, label = "error < 1 %")
plt.plot(num_dps_left, cen_error_5, label = "error < 5 %")
plt.title("percentage of test cases, whose predicted center has an error smaller than a given value", fontsize = 20)
plt.legend(prop = {'size' : 20})
plt.tick_params(axis = 'both', labelsize = 20)
plt.xlabel("set center", fontsize = 20)
plt.ylabel('percentage', fontsize = 20)
plt.show()

plt.plot(num_dps_left, amp_error_1, label = "error < 1 %")
plt.plot(num_dps_left, amp_error_5, label = "error < 5 %")
plt.title("percentage of test cases, whose predicted amplitude has an error smaller than a given value", fontsize = 20)
plt.legend(prop = {'size' : 20})
plt.tick_params(axis = 'both', labelsize = 20)
plt.xlabel("set center", fontsize = 20)
plt.ylabel('percentage', fontsize = 20)
plt.show()

center_selected = np.array([center[i] for i in [3, 6, 9, 21, 37, 36, 34, 28, 17, 19, 24, 26]])
center_selected.sort()
fig, axes = plt.subplots(4, 3)
for i in range(len(center_selected)):
    ind = np.where(center == center_selected[i])[0]
    axes[i // 3, i % 3].plot(result_logistic_function[ind,:,3], result_logistic_function[ind,:,1], '.', color = 'green')
    axes[i // 3, i % 3].plot([center_selected[i]], [2], 'or')
    axes[i // 3, i % 3].set_title('number of data points on the left side of step: ' + str(int(num_dps_left[ind])))
    axes[i // 3, i % 3].set_xlim([1,2])
    axes[i // 3, i % 3].xaxis.set_visible(False)
    if (i //3 == 3):
        axes[i // 3, i % 3].xaxis.set_visible(True)
    axes[i // 3, i % 3].tick_params(axis = 'both', labelsize = 12)
plt.show()

sys.exit()


#performance of convolution at different center
for i in range(len(result_convolve)):
    plt.plot(result_convolve[i,:,0], result_convolve[i,:,1], '.', label = 'found step for each test case')
    plt.plot([center[i]], [2], 'or', label = 'set step')
    plt.xlabel('found center', fontsize = 20)
    plt.ylabel('found amplitude', fontsize = 20)
    plt.title('Distribution of found step for each test case at set center: ' + str(center[i]), fontsize = 20)
    plt.tick_params(axis = 'both', labelsize = 20)
    plt.legend(prop = {'size':20})
    plt.show()







