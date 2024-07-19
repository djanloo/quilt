import numpy as np
from quilt.builder import OscillatorNetwork
from quilt.builder import EEGcap

import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler

global_coupling = 0.01
conduction_speed = 1.0
net = OscillatorNetwork.homogeneous_from_TVB('brain_data/connectivity_76.zip', 
                                             {'oscillator_type':'jansen-rit'}, 
                                             global_weight=global_coupling, 
                                             conduction_speed=conduction_speed)
net.build()

T = 5000
np.random.seed(1998)
states = np.random.uniform(0, 0.05, size=6*net.n_oscillators).reshape(net.n_oscillators, 6)
net.initialize(states, dt=1)
net.run(time=T)

fig,axes = plt.subplots(3,2)
for name, number in zip(net.oscillators.keys(), range(2)):
    for i, ax in enumerate(axes.flatten()):
        ax.plot(net.oscillators[name].history[:, i], label=name)

plt.xlabel("t [ms]")
plt.legend()
plt.suptitle(f"Global coupling strength = {global_coupling:.2f}")

plt.figure(2)
cap = EEGcap("brain_data/regionMapping_16k_76.txt", "brain_data/projection_eeg_65_surface_16k.npy")
plt.matshow(cap.weights)
plt.colorbar()

# plt.imshow(cap.eeg_gain)
N = 20
fig, axes = plt.subplots(N,1, sharex=True)
eeg = cap.eeg(net, filter_signal=True)
for i in range(N):
    axes[i].plot(eeg[i]/np.max(np.abs(eeg[i])) + i + 0.4)
plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# positions = np.array(list(net.features['centers'].values()))
# ax.scatter(*positions.T)

# burn_in = 2500
# for name_x, _ in zip(net.oscillators, range(5)):
#     for name_y , _ in zip(net.oscillators, range(5)):
#         x = net.oscillators[name_x].history[burn_in:, 0]
#         y = net.oscillators[name_y].history[burn_in:, 0]

#         cross_corr = np.correlate(x, y, mode='full')
#         coherence = np.abs(np.fft.fftshift(np.fft.ifft(np.fft.fft(x) * np.conj(np.fft.fft(y)))))

#     plt.figure(figsize=(12, 4))
#     plt.subplot(1, 2, 1)
#     plt.plot(cross_corr)
#     plt.title('Cross-Correlazione')
#     plt.subplot(1, 2, 2)
#     plt.plot(coherence)
#     plt.title('Coerenza')
plt.show()
