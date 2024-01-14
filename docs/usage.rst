Usage
=====

with Python/Jupyter
-------------------
Configuration files are written in yaml.

Neuron catalogues
^^^^^^^^^^^^^^^^^
A catalogue is a list of neuron models:

.. code-block:: yaml

    # neuron_catalogue.yml

    D1_spiny:
        neuron_type: aeif
        C_m : 40.
        tau_m : 200.0

        E_rest: -70.0
        E_reset: -55.0
        E_thr: 0.0
        tau_refrac: 0.0

        # Exp pars
        Delta:  2
        exp_threshold: -40

        # Adapting pars
        ada_a: 0.0
        ada_b: 5.0
        ada_tau_w: 100.0

        # Syn pars
        tau_e: 10.
        tau_i: 5.5
        E_exc: 0.
        E_inh: -65.

To load a neuron catalogue:

.. code-block:: python

    from quilt.builder import NeuronCatalogue
    catalogue = NeuronCatalogue.from_yaml("neuron_catalogue.yml")

Spiking networks
^^^^^^^^^^^^^^^^

Spiking networks are defined by a list of populations and projections. 
Each population uses a model defined in a catalogue.

.. code-block:: yaml
   
    # network_model.yml
    # This is a nonsense example network
    
    populations:
      - name: STR1
        size: 3000
        neuron_model: D1_spiny
      - name: GPi
        size: 3000
        neuron_model: gpi_version3

    projections:
      - name: GPi->STR1
        efferent: GPi
        afferent: STR1
        features:
            exc_fraction: 0.1
            max_exc: 0.5
            min_delay: 0.1
      - name: STR1->GPi
        efferent: STR1
        afferent: GPi
        features:
            inh_fraction: 0.1
            max_inh: 0.03

To build a spiking network:

.. code-block:: python

    from quilt.builder import SpikingNetwork
    spikenet = SpikingNetwork.from_yaml("network_model.yml", catalogue)


in pure C++
-----------

TODO

