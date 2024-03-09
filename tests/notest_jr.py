import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def sigm(v, nu_max,v0,r): 
    result = nu_max / (1.0 + np.exp(r*(v0-v)))
    return result


def f(y, t):
    A = 3.25
    B = 22
    a = 100.0
    b = 50.0

    vmax = 5
    v0 = 6
    C = 135
    r = 0.56

    ff = np.zeros(6)
    ff[0] = y[3]
    ff[1] = y[4]
    ff[2] = y[5]

    ff[3] = A*a*sigm( y[1] - y[2], vmax, v0, r) - 2*a*y[3] - a*a*y[0]
    ff[4] =  A*a*(  200 + 0.8*C*sigm(C*y[0], vmax, v0, r) ) - 2*a*y[4] - a*a*y[1]
    ff[5] =  B*b*0.25*C*sigm(0.25*C*y[0], vmax, v0, r) - 2*b*y[5] - b*b*y[2]
    return ff
h = 0.1
N = 100
y0 = np.array([  0.13,  23.9,  16.2,  -0.14,   5.68, 108.2])
time = np.linspace(0,N*h,N, endpoint=False)
sol = odeint(f, y0, time)
print(sol[-1])
plt.plot(time, sol)


func = np.zeros((len(sol), 6))
for i in range(len(sol)):
    func[i] = f(sol[i], 0)
plt.plot(time, func[:,0])
plt.show()