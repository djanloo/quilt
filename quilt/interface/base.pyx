# distutils: language = c++
import numpy as np
cimport numpy as np

from libcpp.vector cimport vector

# TODO: solve relative import from quilt.interface import cinterface
from quilt.interface.cinterface cimport ParaMap as cParaMap
from quilt.interface.cinterface cimport set_verbosity as cset_verbosity

cpdef set_verbosity(value):
    cset_verbosity(value)

cdef class ParaMap:

    def __cinit__(self, dict params):
        self.params_dict = params.copy()
        self._paramap = new cParaMap()

        for key in self.params_dict.keys():
            key_bytes = key.encode('utf-8') if isinstance(key, str) else key
            if isinstance(self.params_dict[key], int):
                self._paramap.add_float(key_bytes, float(self.params_dict[key]))
            elif isinstance(self.params_dict[key], float):
                self._paramap.add_float(key_bytes, <float>self.params_dict[key])
            elif isinstance(self.params_dict[key], str):
                self._paramap.add_string(key_bytes, self.params_dict[key].encode('utf-8'))
            else:
                raise ValueError(f"Value not accepted in ParaMap construction: {key}->{self.params_dict[key]}")

    
    def __dealloc__(self):
        if self._paramap != NULL:
            del self._paramap
    
    def __reduce__(self):
        return (self.__class__, (self.params_dict,))


cdef class ParaMapList:

    def __cinit__(self):
        pass
    
    def add(self, dict params_dict):
        cdef cinter.ParaMap * new_paramap = new cinter.ParaMap()

        for key in self.params_dict.keys():
            key_bytes = key.encode('utf-8') if isinstance(key, str) else key
            if isinstance(self.params_dict[key], int):
                self._paramap.add_float(key_bytes, float(self.params_dict[key]))
            elif isinstance(self.params_dict[key], float):
                self._paramap.add_float(key_bytes, <float>self.params_dict[key])
            elif isinstance(self.params_dict[key], str):
                self._paramap.add_string(key_bytes, self.params_dict[key].encode('utf-8'))
            else:
                raise ValueError(f"Value not accepted in ParaMap construction: {key}->{self.params_dict[key]}")

        self.paramap_vector.push_back(new_paramap)
    
    def load_list(self, list parameters):
        for par in parameters:
            self.add(par)

    def __dealloc__(self):
        for i in range(self.paramap_vector.size()):
            if self.paramap_vector[i] != NULL:
                del self.paramap_vector[i]
        
    def __reduce__(self):
        return (self.__class__, (self.params_dict,))



cdef class Projection:
    def __cinit__(self,  np.ndarray[np.float32_t, ndim=2,mode='c'] weights, np.ndarray[np.float32_t,ndim=2,mode='c'] delays):
        # print("Making projection")
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
        
        self._projection = new cinter.Projection(self._weights, self._delays)
        # print("Projection done")
        

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

