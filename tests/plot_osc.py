import numpy as np 
import matplotlib.pyplot as plt 

def sigm(v, nu_max,v0,r): 
    result = nu_max / (1.0 + np.exp(r*(v0-v)))
    return result

u = np.loadtxt("output.txt")
conversion = np.array([1,1,1, 1e3, 1e3, 1e3, 1,1,1, 1e3,1e3,1e3])
conversion = np.ones(12)
print(f"shape  {u.shape}" )
tt = np.arange(0, len(u))
for i, uu in enumerate(u.T):
    plt.plot(tt, uu, label=i)
plt.legend()


plt.figure(2)
plt.plot(u[:, 2] - u[:, 3])

plt.show()