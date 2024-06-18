"""
Python snippet for Inhomogeneous Poisson process generator
"""
import numpy as np
import matplotlib.pyplot as plt

def rate_of_spikes(spikes, time_bins, N_samples):

    # Time in ms

    N_bins = len(time_bins) - 1
    rate = time_bins*0
    for i in range(N_bins):
        mask = (spikes > time_bins[i])&(spikes <= time_bins[i+1])
        rate[i] = np.sum(mask)/(time_bins[i+1] - time_bins[i])/N_samples

    return rate*1000


def gen_inh_poiss(time, _rate_function):

    # Time in millis
    # Rate in Hz
    # So rate must be converted in ms^{-1}
    
    rate_function = _rate_function/1e3

    dt = np.diff(time)[0]
    last_spike_timebin = 0
    spike_times = []
    Y = 0
    timebin_passed = 0
    abort_flag = False
    while timebin_passed < len(time):
        y = -np.log(np.random.uniform(0,1))
        # print(f"y = {y}")
        while True:
            if last_spike_timebin + timebin_passed >= len(time) - 1 :
                abort_flag = True
                break
            Y += (rate_function[last_spike_timebin + timebin_passed] + rate_function[last_spike_timebin + timebin_passed + 1])*dt/2
            # Y += rate_function[last_spike_timebin + timebin_passed]*dt

            timebin_passed += 1
            # print(f"\tlast_spike_timebin:{last_spike_timebin}, timebin: {timebin_passed}, Y: {Y}")
            if Y >= y:
                break
        if abort_flag:
            # print(f"Process was aborted because:\nlast_spike_timebin: {last_spike_timebin} and timebin_passed: {timebin_passed}")
            break
        # print("Y exceeded y_i")
        # print(f"Spike happened at timebin {timebin_passed} (t = {timebin_passed*dt})")
        last_spike_timebin += timebin_passed
        timebin_passed = 0
        Y = 0
        spike_times.append(last_spike_timebin*dt)
    
    return spike_times
    

T = 1000  # ms: max time
dt = 0.1 # ms: timestep

ampl_f = 200    # Hz: amplitude of rate oscillation
offset_f = 200  # Hz: offset of rate oscillation
nu_0 = 10/2       # Hz: frequency of rate oscllation

t = np.arange(0, T/dt)*dt # ms: time
t_sec = t*1e-3
rho = ampl_f*np.sin(2*np.pi*nu_0 * t_sec)**2 + offset_f  # Hz: rate function

ensamble_spike_times = []

N_samples = 5000
for i in range(N_samples):
    print(i, end="-", flush=True)
    spike_times = gen_inh_poiss(t, rho)
    ensamble_spike_times += spike_times

# print(ensamble_spike_times)

ensamble_spike_times = np.array(ensamble_spike_times)
plt.step(t[::10][:-1], rate_of_spikes(ensamble_spike_times, t[::10], N_samples)[:-1], where="post")


plt.plot(t, rho)
plt.xlabel("time [ms]")
plt.ylabel("Rate [Hz]")
plt.show()