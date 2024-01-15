from libcpp cimport bool
cimport quilt.interface.cinterface as cinter

cdef class ParaMap:
    cdef:
        cinter.ParaMap * _paramap
        bool is_neuron_paramap
        bool is_oscillator_paramap