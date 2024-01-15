# distutils: language = c++
import numpy as np
cimport numpy as np

from quilt.src_cython.cinterface cimport ParaMap as cParaMap
from quilt.src_cython.cinterface cimport NEURON_TYPES

cdef class ParaMap:

    def __cinit__(self, dict params):
        self._paramap = new cParaMap()

        self.is_neuron_paramap = "neuron_type" not in params.keys()
        self.is_oscillator_paramap = "oscillator_type" not in params.keys()

        if (not self.is_neuron_paramap) and (not self.is_oscillator_paramap):
            message = "ParaMap must have a 'neuron_type' or 'oscillator_type' field\n"
            message += f"Possible values are:\n"
            message += f"\tneuron_type: {list(NEURON_TYPES.keys())}"
            # message += f"\oscillator_type: {list(OSCILLATOR_TYPES.keys())}"

            raise KeyError(message)
        
        if self.is_neuron_paramap and self.is_oscillator_paramap:
            raise ValueError("ParaMap cannot belong to neuron and oscillator")

        # Converts to unsigned int
        if "neuron_type" in params.keys():
            params['neuron_type'] = NEURON_TYPES[params['neuron_type']]
            
        # Converts to unsigned int
        # if "oscillator_type" in params.keys():
        #     params['oscillator_type'] = OSCILLATOR_TYPES[params['oscillator_type']]
    
        for key in params.keys():
            key_bytes = key.encode('utf-8') if isinstance(key, str) else key
            self._paramap.add(key_bytes, params[key])
