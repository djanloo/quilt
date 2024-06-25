import numpy as np
import matplotlib.pyplot as plt


osc_vars = np.loadtxt("cortex.txt")
N_oscill = osc_vars.shape[1]
for i in range(N_oscill):
    plt.plot(osc_vars[:, i])
plt.show()