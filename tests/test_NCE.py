import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

plt.style.use('style.mplstyle')
# Checks for NCEs in y' = f(t, y)

def f(y, t):
    return -y + np.sin(10*t)

def exact(t):
    return 111/101*np.exp(-t) + 1/101*np.sin(10*t) - 10/101*np.cos(10*t)

cs = np.array([0,0.5, 0.5, 1])
bs = np.array([1/6,1/3,1/3,1/6])
a = np.array([0, 0.5, 0.5, 1])

def b_theta(theta, i):
    global bs, cs
    if i == 0:
        return 2*(1-4*bs[0])*theta**3 + 3*(3*bs[0] - 1)*theta**2 + theta
    else:
        return 4*(3*cs[i] - 2)*bs[i]*theta**3 + 3*(3-4*cs[i])*bs[i]*theta**2

N = 100
h = 0.1

K = np.zeros((N, 4))
y = np.zeros(N)
tt = np.linspace(0, N*h, N, endpoint=False)

y[0] = 1


for i in range(N):
    for nu in range(4):
        t_eval = (i+cs[nu])*h
        y_eval = y[i]
        if nu > 0:
            y_eval += h*a[nu]*K[i, nu-1]

        K[i, nu] = f(y_eval, t_eval)

    if i + 1 < N:
        y[i+1] = y[i]
        for nu in range(4):
            y[i+1] += h*bs[nu]*K[i, nu]

fig, ax = plt.subplots(constrained_layout=True)
plt.plot(tt, y, ls="", marker='.', color="k", label=f"RK4")

M = 10*N
tt_continuous = np.linspace(0, N*h, M, endpoint=False)
y_continuous = np.zeros(M)

for i in range(M):
    cell_id = int(np.floor(tt_continuous[i]/h))
    theta = tt_continuous[i]/h - cell_id
    print(f"i = {i}, time = {tt_continuous[i]}, cell_id = {cell_id}, theta = {theta}")
    continuous_delta = 0
    for nu in range(4):
        continuous_delta += h* b_theta(theta, nu) * K[cell_id, nu]
        print(f"nu = {nu}, theta = {theta}, b_theta = {b_theta(theta, nu)}")
    print(f"Delta = {continuous_delta}")
    y_continuous[i] = y[cell_id] + continuous_delta

plt.plot(tt_continuous, y_continuous, ls="--", color="r", label="NCE", zorder=5)


sol = odeint(f, y[0], tt_continuous)
plt.plot(tt_continuous, exact(tt_continuous), color="k", alpha=0.8, label=f"Exact")
plt.legend()
plt.xlabel("t")
plt.ylabel("y")

t1, t2, y1, y2 = 8.38, 8.55, 0.07, 0.11  # subregion of the original image
mask = (tt_continuous>t1)&(tt_continuous < t2)
axins = ax.inset_axes(
    [0.2, 0.5, 0.47, 0.47],
    xlim=(t1, t2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
axins.plot(tt_continuous, y_continuous, ls="", marker=".", ms=10, color="r", label="NCE", zorder=10)
axins.plot(tt, y, ls="", marker='o', color="k", ms=10)
axins.plot(tt_continuous, exact(tt_continuous), color="k", alpha=0.8)
ax.indicate_inset_zoom(axins, edgecolor="black")


plt.figure(2)
thetas = np.linspace(0, 1, 100)
sum_coeff = np.zeros(len(thetas))
for nu in range(4):
    coeff = b_theta(thetas, nu)
    plt.plot(thetas, coeff, label = f"$b_{nu}$")
    sum_coeff += coeff
plt.plot(thetas, sum_coeff, label="sum")
plt.legend()
plt.show()