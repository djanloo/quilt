# distutils: language = c++
import numpy as np
cimport numpy as np
import ctypes


# from libc import free, malloc
from libcpp cimport vector

# TODO: solve relative import from quilt.interface import cinterface
from quilt.interface.cinterface cimport ParaMap as cParaMap

# TODO: import these as extern from C++ file
NEURON_TYPES = {"base_neuron":0, "aqif":1,"aqif2":2 ,"izhikevich":3, "aeif":4}
OSCILLATOR_TYPES = {"base_oscillator":0, "harmonic": 1}

cdef class ParaMap:

    def __cinit__(self, dict params):
        self.params_dict = params.copy()
        self.converted_params_dict = self.params_dict.copy()
        self._paramap = new cParaMap()

        self.is_neuron_paramap = ("neuron_type" in self.params_dict.keys())
        self.is_oscillator_paramap = ("oscillator_type" in self.params_dict.keys())

        if (not self.is_neuron_paramap) and (not self.is_oscillator_paramap):
            message = "ParaMap must have a 'neuron_type' or 'oscillator_type' field\n"
            message += f"Possible values are:\n"
            message += f"\tneuron_type: {list(NEURON_TYPES.keys())}"
            message += rf"\oscillator_type: {list(OSCILLATOR_TYPES.keys())}"

            raise KeyError(message)
        
        if self.is_neuron_paramap and self.is_oscillator_paramap:
            raise ValueError("ParaMap cannot belong to neuron and oscillator")

        # Converts to unsigned int
        if "neuron_type" in self.params_dict.keys():
            self.converted_params_dict['neuron_type'] = NEURON_TYPES[self.params_dict['neuron_type']]
            
        # Converts to unsigned int
        if "oscillator_type" in self.params_dict.keys():
            self.converted_params_dict['oscillator_type'] = OSCILLATOR_TYPES[self.params_dict['oscillator_type']]
    
        for key in self.converted_params_dict.keys():
            key_bytes = key.encode('utf-8') if isinstance(key, str) else key
            self._paramap.add(key_bytes, self.converted_params_dict[key])
    
    def __dealloc__(self):
        if self._paramap != NULL:
            del self._paramap
    
    def __reduce__(self):
        return (self.__class__, (self.params_dict,))


cdef class Projection:
    def __cinit__(self,  np.ndarray[np.float32_t, ndim=2,mode='c'] weights, np.ndarray[np.float32_t,ndim=2,mode='c'] delays):

        self.weights = weights
        self.delays = delays

        self.start_dimension = weights.shape[0]
        self.end_dimension   = weights.shape[1]

        if delays.shape[0] != weights.shape[0]:
            raise ValueError(f"Weights and delays have different start dimension: {weights.shape[0]} vs {delays.shape[0]}")
        if delays.shape[1] != weights.shape[1]:
            raise ValueError(f"Weights and delays have different end dimension: {weights.shape[1]} vs {delays.shape[1]}")

        self._weights = vector[vector[float]](self.start_dimension)
        self._delays = vector[vector[float]](self.start_dimension)

        for i in range(self.start_dimension):
            weights_row = vector[float](self.end_dimension)
            delays_row = vector[float](self.end_dimension)

            for j in range(self.end_dimension):

                weights_row[j] = weights[i,j]
                delays_row[j] = delays[i,j]

            self._weights[i] = weights_row
            self._delays[i] = delays_row
        

    @property
    def weights(self):
        return self.weights
    
    @property
    def delays(self):
        return self.delays

    @property
    def cweights(self):
        return self._weights
    
    @property
    def cdelays(self):
        return self._delays

