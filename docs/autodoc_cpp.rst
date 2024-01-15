C++ code references
===================

Base spiking dynamics
---------------------
.. doxygenfile:: quilt/src_cpp/include/neurons_base.hpp
   :project: quilt

Neuron models
-------------

.. doxygenfile:: quilt/src_cpp/include/neuron_models.hpp
   :project: quilt


Oscillators
-----------

.. warning::
   
   This section is under construction. The code is not well defined yet.

A network of oscillators is defined by a matrix of weights :math:`W_{ij}` and a matrix of delays :math:`\Delta_{ij}`.

.. doxygenfile:: quilt/src_cpp/include/oscillators.hpp
   :project: quilt