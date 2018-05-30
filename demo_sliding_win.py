from logistic_function import *
import sys

num_dps = 1000
center_set = 1.5
step_set = 2
ws = 100

x = np.linspace(1, 2, num_dps)
np.random.seed(21567862)
y = np.random.standard_normal(num_dps)
y[x > center_set] += step_set
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
ps = moving_median(ps_eff, 49)
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
figi, gs, axes = plot_combi(x, y, ps, cen, amp, bins, "position", "center", "amplitude", fig_title = "center: "+str(center_set)+ ", amplitude: " + str(step_set))
ax = figi.add_subplot(gs[3:5, 1:3])
X, Y = np.meshgrid(cen_edge, amp_edge)
ax.set_xlabel("center calculated", fontsize = 20)
ax.set_ylabel("amplitude calculated", fontsize = 20)
histo = ax.pcolormesh(X, Y, H)
ax1 = figi.add_subplot(gs[3:5, 3])
divider = make_axes_locatable(ax1)
cax = divider.append_axes("left", size = "5%", pad = 0.08)
figi.colorbar(histo, cax = cax)
ax1.axis("off")
plt.tick_params(axis = 'both', labelsize = 20)
ax.plot(cen, amp, "og")
ax.tick_params(axis = 'both', labelsize = 20)
figi.subplots_adjust(left=0.01, bottom=0.1, right=0.99, top=0.95,wspace=0.3, hspace=0.1)
plt.show()
