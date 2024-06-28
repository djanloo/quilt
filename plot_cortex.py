import numpy as np
import matplotlib.pyplot as plt


osc_vars = np.loadtxt("cortex.txt")
N_oscill = osc_vars.shape[1]

for i in range(N_oscill):
    plt.plot(osc_vars[:, i], marker=".")

osc_vars_interp = np.loadtxt("cortex_interpolated.txt")
print(osc_vars_interp)
for i in range(N_oscill):
    plt.plot(np.arange(len(osc_vars_interp[:, i])) *0.1, osc_vars_interp[:, i], marker=".", ms=4)

plt.show()