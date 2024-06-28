import numpy as np
import matplotlib.pyplot as plt

dT = 2
dt = 0.05

osc_vars = np.loadtxt("cortex.txt")
N_oscill = osc_vars.shape[1]

for i in range(N_oscill):
    plt.plot(np.arange(len(osc_vars[:, i]))*dT, osc_vars[:, i], marker=".")

osc_vars_interp = np.loadtxt("cortex_interpolated.txt")
print(osc_vars_interp)
for i in range(N_oscill):
    plt.plot(np.arange(len(osc_vars_interp[:, i])) *dt, osc_vars_interp[:, i], marker=".", ms=4)

plt.show()