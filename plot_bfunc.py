import numpy as np
import matplotlib.pyplot as plt

tb = np.loadtxt("b_functions.txt")

for i in range(4):
    plt.plot(tb[:, 0], tb[:, 1 +i], label=f"{i}")
plt.show()