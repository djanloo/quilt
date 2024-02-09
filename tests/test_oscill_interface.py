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

    w = np.arange(16).reshape((4,4)).astype(np.float32)
    d = w.copy()

    proj = base_objects.Projection(w,d)
    net.build_connections(proj)


if __name__=="__main__":
    test_build_connections()