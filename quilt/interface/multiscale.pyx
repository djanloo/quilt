cimport quilt.interface.cinterface as cinter
cimport quilt.interface.base as base

cimport quilt.interface.spiking as spiking
cimport quilt.interface.oscill as oscill

cdef class Transducer:
    cdef cinter.Transducer * _transducer

    def __cinit__(self, spiking.Population population, base.ParaMap params, MultiscaleNetwork multinet):
        self._transducer = new cinter.Transducer(population._population, params._paramap, multinet._multiscale_network)

cdef class MultiscaleNetwork:
    cdef cinter.MultiscaleNetwork * _multiscale_network 
    cdef cinter.EvolutionContext * _evo_short
    cdef cinter.EvolutionContext * _evo_long

    def __cinit__(self, spiking.SpikingNetwork spikenet, oscill.OscillatorNetwork oscnet):
        self._multiscale_network = new cinter.MultiscaleNetwork(spikenet._spiking_network, oscnet._oscillator_network)

    def __init__(self):
        self.transducers = []

    def add_transducer(self, spiking.Population population, base.ParaMap params):
        self.transducers.append(Transducer(population, params, self))

    # def connect_scales(self,  base.Projection T2O, base.Projection O2T):
    #     cdef int i,j
    #     for i in range():
    #     self._multiscale_network.build_OT_projections(T2O, O2T)