import numpy as np 
import matplotlib.pyplot as plt 

u = np.loadtxt("output.txt")

plt.plot(u[:,0])
# plt.plot(u[:,1])

plt.plot(u[:,2])
# plt.plot(u[:,3])

plt.plot(u[:,4])
# plt.plot(u[:,1])

plt.plot(u[:,6])
# plt.plot(u[:,3])

plt.figure(2)

plt.plot(u[:, 0]**1 + u[:, 1]**2)

plt.show()