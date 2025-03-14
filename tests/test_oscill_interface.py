from quilt.interface import oscill
from quilt.interface import base
from quilt.builder import OscillatorNetwork

import numpy as np

def test_oscnet():
    params = base.ParaMap({'oscillator_type': "jansen-rit"})
    net = oscill.OscillatorNetwork.homogeneous(3, params)

def test_build_connections():
    
    N = 3
    params = base.ParaMap({'oscillator_type': "jansen-rit"})
    net = oscill.OscillatorNetwork.homogeneous(N, params)

    w = np.arange(N*N).reshape((N,N)).astype(np.float32) + 1
    d = w.copy()

    proj = base.Projection(w,d)

    link_params = base.ParaMap({})
    net.build_connections(proj, link_params)
        

def test_run():
    N = 3
    params = base.ParaMap({'oscillator_type': "jansen-rit"})
    net = oscill.OscillatorNetwork.homogeneous(N, params)

    w = np.arange(N*N).reshape((N,N)).astype(np.float32) + 2
    d = w.copy()
    print(d)

    proj = base.Projection(w,d)

    link_params = base.ParaMap({})
    net.build_connections(proj, link_params)

    net.initialize(tau=1, vmin=0.0, vmax=0.25, dt=1)

    net.run(time=100)

def test_tvb():
    global_coupling = 0.01
    conduction_speed = 1.0
    net = OscillatorNetwork.homogeneous_from_TVB('tests/connectivity_76.zip', 
                                                {'oscillator_type':'jansen-rit'}, 
                                                global_weight=global_coupling, 
                                                conduction_speed=conduction_speed)
    net.build()
    net.initialize(tau=1, vmin=0.0, vmax=0.25, dt=1)
    net.run(time=100)

def test_history():
    global_coupling = 0.01
    conduction_speed = 1.0
    net = OscillatorNetwork.homogeneous_from_TVB('tests/connectivity_76.zip', 
                                                {'oscillator_type':'jansen-rit'}, 
                                                global_weight=global_coupling, 
                                                conduction_speed=conduction_speed)
    net.build()
    net.initialize(tau=1, vmin=0.0, vmax=0.25, dt=1)
    net.run(time=100)
    net.oscillators['lTCV'].history


def test_rate_history():
    global_coupling = 0.01
    conduction_speed = 1.0
    net = OscillatorNetwork.homogeneous_from_TVB('tests/connectivity_76.zip', 
                                                {'oscillator_type':'jansen-rit'}, 
                                                global_weight=global_coupling, 
                                                conduction_speed=conduction_speed)
    net.build()
    net.initialize(tau=1, vmin=0.0, vmax=0.25, dt=1)
    net.run(time=100)
    net.oscillators['lTCV'].rate_history



if __name__=="__main__":
    test_rate_history()
    pass