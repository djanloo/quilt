import numpy as np 
import matplotlib.pyplot as plt 

u = np.loadtxt("output.txt")
plt.plot(u[:,0])
# plt.plot(u[:,1])

plt.plot(u[:,2])
# plt.plot(u[:,3])

plt.show()