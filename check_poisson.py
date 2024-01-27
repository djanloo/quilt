import numpy as np 
import matplotlib.pyplot as plt 
# TMAX = 50
p, n, t = np.loadtxt("spikes.txt", unpack=True)
# print(p)
# mask = (t<TMAX)
# n = n[mask]
# t = t[mask]
# p = p[mask]

t = t[p==0]
n = n[p==0]

for i in range(4):
    u = t[n==i]
    print(f"AVG tau = {np.mean(np.diff(u))}")
    a,b = np.histogram(np.diff(u), bins=30)
    centers = b[:-1] + 0.5 * np.diff(b)
    plt.plot(centers, a)

    taus = centers * np.log(a)
    tau = np.mean(taus[np.isfinite(taus)])
    print(tau)
plt.yscale('log')
plt.show()