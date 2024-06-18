import numpy as np
import matplotlib.pyplot as plt

n, tsp = np.loadtxt("test_inh_poiss.txt", unpack=True)

plt.hist(tsp, bins=100)
plt.show()