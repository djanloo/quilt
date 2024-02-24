import numpy as np 
import matplotlib.pyplot as plt 

def sigm(v, nu_max,v0,r): 
    result = nu_max / (1.0 + np.exp(r*(v0-v)))
    return result

u = np.loadtxt("output.txt")
conversion = np.array([1,1,1, 1e3, 1e3, 1e3])
print(f"shape  {u.shape}" )
tt = np.arange(0, len(u))*0.1
for uu, fact in zip(u.T, conversion):
    
    plt.plot(tt, sigm(fact*uu, 5,6, 0.56))

# u = np.diff(u)/np.diff(tt)
# tt = tt[:-1]
# plt.plot(tt, u)

# u = np.diff(u)/np.diff(tt)
# tt = tt[:-1]
# plt.plot(tt, u)

# u = np.diff(u)/np.diff(tt)
# tt = tt[:-1]
# plt.plot(tt, u)

# plt.yscale('log')
# plt.xscale('log')
plt.show()