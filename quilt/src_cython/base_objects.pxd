cimport quilt.src_cython.cinterface as cinter

cdef class ParaMap:
    cdef cinter.ParaMap * _paramap
