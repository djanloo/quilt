import numpy as np 
import matplotlib.pyplot as plt 
from scipy.sparse import coo_matrix

i,j,w,d = np.loadtxt("output.txt", unpack=True)
i = i.astype(int)
j = j.astype(int)

sparse_m = coo_matrix((d, (i ,j)))

# plt.matshow(sparse_m.toarray())
bins = 1000
occ_w = plt.hist(w, bins=bins, histtype="step", label="weight")[0]
occ_d = plt.hist(d, bins=bins, histtype="step", label ="delay")[0]

max_val = max(np.max(occ_w), np.max(occ_d))

plt.annotate(f"weight={np.mean(w):.1f} +- {np.std(w):.2f}", xy=(np.mean(w)*1.2, max_val*(1-0.1)))
plt.annotate(f"delay={np.mean(d):.1f} +- {np.std(d):.2f}", xy=(np.mean(d)*1.2, max_val*(1-0.15)))

plt.ylim(0,max_val*(1+0.1))
plt.legend()
plt.show()