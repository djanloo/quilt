# distutils: language = c++
import numpy as np
cimport numpy as np

# TODO: solve relative import from quilt.interface import cinterface
from quilt.interface.cinterface cimport ParaMap as cParaMap

NEURON_TYPES = {"base_neuron":0, "aqif":1,"aqif2":2 ,"izhikevich":3, "aeif":4}

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
            # message += rf"\oscillator_type: {list(OSCILLATOR_TYPES.keys())}"

            raise KeyError(message)
        
        if self.is_neuron_paramap and self.is_oscillator_paramap:
            raise ValueError("ParaMap cannot belong to neuron and oscillator")

        # Converts to unsigned int
        if "neuron_type" in self.params_dict.keys():
            self.converted_params_dict['neuron_type'] = NEURON_TYPES[self.params_dict['neuron_type']]
            
        # Converts to unsigned int
        # if "oscillator_type" in params_dict.keys():
        #     params_dict['oscillator_type'] = OSCILLATOR_TYPES[params_dict['oscillator_type']]
    
        for key in self.converted_params_dict.keys():
            key_bytes = key.encode('utf-8') if isinstance(key, str) else key
            self._paramap.add(key_bytes, self.converted_params_dict[key])
    
    def __dealloc__(self):
        if self._paramap != NULL:
            del self._paramap
    
    def __reduce__(self):
        return (self.__class__, (self.params_dict,))
