from quilt.interface import oscill
from quilt.interface import base
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

    proj = base.Projection(w,d)
    net.build_connections(proj)
        
    return net

def test_run():
    
    net = test_build_connections()
    net.init(np.zeros((4, 2)))
    net.run(time=100)

def test_homogeneous_builder():
    from quilt.builder import OscillatorNetwork
    N = 2
    params = dict(oscillator_type='jansen-rit')
    w = np.ones((N,N))
    for i in range(N):
        w[i,i] = 0
    d = np.ones((N,N))*0
    net = OscillatorNetwork.homogeneous(params, w, d)
    net.build()
    init_cond = np.array([[0.13, 20.0, 20.0, 1e-6, 1e-6, 1e-6],[0.13, 20.0, 20.0, 1e-6, 1e-6, 1e-6]])
    net.init(init_cond, dt=0.1)
    net.run(dt=1,time=1000)
    return net

def test_get_history():
    net = test_homogeneous_builder()
    for osc in net.oscillators:
        net.oscillators[osc].history
    return net

def test_connect():
    from quilt.builder import OscillatorNetwork
    N = 2
    params = dict(oscillator_type='jansen-rit')
    w =  np.random.normal(0.5, 0.1, size=(N,N))
    for i in range(N):
        w[i,i] = 0
    d = np.random.normal(1, 0.1, size=(N,N))
    net = OscillatorNetwork.homogeneous(params, w, d)
    net.build()
    init_cond = np.array([[0.13, 20.0, 20.0, 1e-6, 1e-6, 1e-6],
                          [0.13, 20.0, 20.0, 1e-6, 1e-6, 1e-6]])
    net.init(init_cond, dt=0.1)
    net.run(dt=1,time=1000)
    return net



if __name__=="__main__":
    net = test_connect()
    import matplotlib.pyplot as plt 
    plt.figure(1)
    for osc in net.oscillators:
        plt.plot(net.oscillators[osc].history[:,0])
    plt.figure(2)
    for osc in net.oscillators:
        plt.plot(net.oscillators[osc].history[:,1])
    plt.show()
