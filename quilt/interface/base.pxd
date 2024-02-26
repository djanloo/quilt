from libcpp cimport bool
from libcpp.vector cimport vector
cimport quilt.interface.cinterface as cinter

cdef class ParaMap:
    cdef:
        cinter.ParaMap * _paramap
        bool is_neuron_paramap
        bool is_oscillator_paramap
        dict params_dict, converted_params_dict

cdef class Projection:
    cdef:
        int start_dimension, end_dimension
        vector[vector[float]] _weights
        vector[vector[float]] _delays 
        cinter.Projection * _projection
        float [:,:] weights, delays
