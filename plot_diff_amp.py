import numpy as np
import scipy.stats
import scipy.signal
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys

#influence of amplitude of the results
step = np.around(np.exp(np.linspace(0, 2, 25)) - 1, decimals = 1)

result_logistic_function = np.load("/home/whatever/PythonProjects/DwellAnalysis/abschluss/data/logistic_function_amp.npy")
result_convolve = np.load("/home/whatever/PythonProjects/DwellAnalysis/abschluss/data/result_convolution_amp.npy")

#robustness of logistic function under different amp
percent_eff_logistic_func = np.zeros(25)

for i in range(len(result_logistic_function)):
    delta = result_logistic_function[i,:,1]
    percent_eff_logistic_func[i] = (delta.size - np.count_nonzero(np.isnan(delta)))/len(delta)

plt.plot(step, percent_eff_logistic_func, '-o')
plt.title('percentage of cases, in which regression parameters can be calculated', fontsize = 20)
plt.xlabel('set amplitude', fontsize = 20)
plt.ylabel('percentage', fontsize = 20)
plt.tick_params(axis = 'both', labelsize = 20)
plt.show()

#how many of the test cases have an error smaller than 1% or 5%
cen_error_1 = np.zeros(len(step) - 1)
amp_error_1 = np.zeros(len(step) - 1)
cen_error_5 = np.zeros(len(step) - 1)
amp_error_5 = np.zeros(len(step) - 1)

for i in range(1, len(result_convolve)):
    center_calc = result_logistic_function[i,:,3]
    center_calc = center_calc[~np.isnan(center_calc)]
    step_calc = result_logistic_function[i,:,1]
    step_calc = step_calc[~np.isnan(step_calc)]
    error_center = abs(center_calc - 1.5) / 1.5
    error_step = abs(step_calc - step[i]) / step[i]
    cen_error_1[i - 1] = len(error_center[error_center < 0.01])/ len(error_center)
    amp_error_1[i - 1] = len(error_step[error_step < 0.01])/ len(error_step)
    cen_error_5[i - 1] = len(error_center[error_center < 0.05])/ len(error_center)
    amp_error_5[i - 1] = len(error_step[error_step < 0.05])/ len(error_step)

plt.plot(step[1:], cen_error_1, label = "error < 1 %")
plt.plot(step[1:], cen_error_5, label = "error < 5 %")
plt.title("percentage of test cases, whose predicted center has an error smaller than a given value", fontsize = 20)
plt.legend(prop = {'size' : 20})
plt.tick_params(axis = 'both', labelsize = 20)
plt.xlabel("set amplitude", fontsize = 20)
plt.ylabel('percentage', fontsize = 20)
plt.show()

plt.plot(step[1:], amp_error_1, label = "error < 1 %")
plt.plot(step[1:], amp_error_5, label = "error < 5 %")
plt.title("percentage of test cases, whose predicted amplitude has an error smaller than a given value", fontsize = 20)
plt.legend(prop = {'size' : 20})
plt.tick_params(axis = 'both', labelsize = 20)
plt.xlabel("set amplitude", fontsize = 20)
plt.ylabel('percentage', fontsize = 20)
plt.show()

#performance of logistic function under different amplitude
for i in range(len(result_convolve)):
    fig, ax1 = plt.subplots()
    ax2 = fig.add_axes(ax1)
    ax1.plot(result_logistic_function[i,:,3], result_logistic_function[i,:,1], '.', label = 'found step for individual test case')
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
    plt.show()

#show some special case at certain amplitude
step_selec = np.array([0, 0.1, 0.3, 0.5, 0.6, 0.8, 1.1, 1.3, 1.7, 2.2, 4.8, 5.3])
fig, axes = plt.subplots(4, 3)
for i in range(len(step_selec)):
    ind = np.where(step == step_selec[i])[0]
    axes[i // 3, i % 3].plot(result_logistic_function[ind,:,3], result_logistic_function[ind,:,1], '.', color = 'green')
    axes[i // 3, i % 3].plot([1.5], [step_selec[i]], 'or')
    axes[i // 3, i % 3].set_title('amplitude: ' + str(step_selec[i]))
    axes[i // 3, i % 3].set_xlim([1,2])
    axes[i // 3, i % 3].xaxis.set_visible(False)
    if (i //3 == 3):
        axes[i // 3, i % 3].xaxis.set_visible(True)
    axes[i // 3, i % 3].tick_params(axis = 'both', labelsize = 12)
plt.show()

#performance of convolution under different amplitude
for i in range(len(result_convolve)):
    plt.plot(result_convolve[i,:,0], result_convolve[i,:,1], '.', label = 'found step for each test case')
    plt.plot([1.5], [step[i]], 'or', label = 'set step')
    plt.xlabel('found center', fontsize = 20)
    plt.ylabel('found amplitude', fontsize = 20)
    plt.xlim([1,2])
    plt.title('Distribution of found step for each test case at set amplitude: ' + str(step[i]), fontsize = 20)
    plt.tick_params(axis = 'both', labelsize = 20)
    plt.legend(prop = {'size':20})
    plt.show()