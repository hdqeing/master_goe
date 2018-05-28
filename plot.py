import numpy as np
import scipy.stats
import scipy.signal
import matplotlib.pyplot as plt
import sys

#influence of center location

center_1 = np.linspace(1, 1.05, 20)
center_2 = np.linspace(1.055, 1.949, 10)
center_3 = np.linspace(1.95, 2, 10)
center = np.hstack([center_1, center_2, center_3])

result_logistic_function = np.load("/home/whatever/PythonProjects/DwellAnalysis/abschluss/data/logistic_function_cen.npy")
result_convolve = np.load("/home/whatever/PythonProjects/DwellAnalysis/abschluss/data/result_convolution_cen.npy")

percent_eff_logistic_func = np.zeros(len(center))

#robustness of logistic function at different center
for i in range(len(result_logistic_function)):
    delta = result_logistic_function[i,:,1]
    percent_eff_logistic_func[i] = (delta.size - np.count_nonzero(np.isnan(delta)))/len(delta)

plt.plot(center, percent_eff_logistic_func, '-o')
plt.title('percentage of cases, where regression parameter can be found', fontsize = 20)
plt.xlabel('set center', fontsize = 20)
plt.ylabel('percentage', fontsize = 20)
plt.tick_params(axis = 'both', labelsize = 20)
plt.show()

#performance of logistic function at different center
for i in range(len(result_convolve)):
    plt.plot(result_logistic_function[i,:,3], result_logistic_function[i,:,1], '.', label = 'found step for each test case')
    plt.plot([center[i]], [2], 'or', label = 'set step')
    plt.xlabel('found center', fontsize = 20)
    plt.ylabel('found amplitude', fontsize = 20)
    plt.title('Distribution of found step for each test case at set center: ' + str(center[i]), fontsize = 20)
    plt.tick_params(axis = 'both', labelsize = 20)
    plt.legend(prop = {'size':20})
    plt.show()

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


sys.exit()

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

#performance of logistic function under different amplitude
for i in range(len(result_convolve)):
    plt.plot(result_logistic_function[i,:,3], result_logistic_function[i,:,1], '.', label = 'found step for individual test case')
    plt.plot([1.5], [step[i]], 'or', label = 'set step')
    plt.xlabel('found center', fontsize = 20)
    plt.ylabel('found amplitude', fontsize = 20)
    plt.xlim([1,2])
    plt.title('Distribution of found step for each test case at set amplitude: ' + str(step[i]), fontsize = 20)
    plt.tick_params(axis = 'both', labelsize = 20)
    plt.legend(prop = {'size':20})
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



