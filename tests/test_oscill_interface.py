from quilt.interface import oscill
from quilt.interface import base_objects
import numpy as np

def test_oscnet():
    net = oscill.OscillatorNetwork()

def test_harmonic():
    params = dict(k=1, x_0=0, v_0=2)
    net = oscill.OscillatorNetwork()
    harm = oscill.harmonic_oscillator(params, net)

def test_build_connections():
    
    params = dict(k=1, x_0=0, v_0=2)
    net = oscill.OscillatorNetwork()

    for i in range(4):
        harm = oscill.harmonic_oscillator(params, net)

    w = np.arange(16).reshape((4,4)).astype(np.float32) + 1
    d = w.copy()

    proj = base_objects.Projection(w,d)
    net.build_connections(proj)
        
    return net

# def test_add_osc():
#     params = dict(k=1, x_0=0, v_0=2)
#     net = oscill.OscillatorNetwork()
#     net.add_oscillator(params)
    

def test_run():
    
    net = test_build_connections()
    net.init(np.zeros((4, 2)))
    net.run(time=100)

def test_homogeneous_builder():
    from quilt.builder import OscillatorNetwork
    N = 2
    params = dict(oscillator_type='jansen-rit', k=1, x_0=1, v_0=1)
    w = np.ones((N,N))
    for i in range(N):
        w[i,i] = 0
    d = np.ones((N,N))*0
    net = OscillatorNetwork.homogeneous(params, w, d)
    net.build()
    net.init(np.arange(N*6).reshape(N, 6).astype(float), dt=0.1)
    net.run(dt=0.1,time=30)
    return net

def test_get_history():
    net = test_homogeneous_builder()
    for osc in net.oscillators:
        net.oscillators[osc].history
    return net

if __name__=="__main__":
    net = test_get_history()
    import matplotlib.pyplot as plt 
    plt.figure(1)
    for osc in net.oscillators:
        plt.plot(net.oscillators[osc].history[:,0])
    plt.figure(2)
    for osc in net.oscillators:
        plt.plot(net.oscillators[osc].history[:,1])
    plt.show()
