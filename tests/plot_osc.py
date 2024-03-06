import numpy as np 
import matplotlib.pyplot as plt 

def sigm(v, nu_max,v0,r): 
    result = nu_max / (1.0 + np.exp(r*(v0-v)))
    return result

u = np.loadtxt("output.txt")
tt = np.arange(0, len(u))

M = 6

for i in range(u.shape[1]//M):
    plt.plot(tt, u[:,M*i + 1] - u[:, M*i + 2], label=f"osc{i}")

plt.legend()


# plt.figure(2)
# plt.plot(u[:, 2] - u[:, 3])

plt.show()