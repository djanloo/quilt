Usage
=====

with Python/Jupyter
-------------------
Configuration files are written in yaml. See also `the example notebook <https://github.com/djanloo/quilt/blob/main/example.ipynb>`_.

Neuron catalogues
^^^^^^^^^^^^^^^^^
A catalogue is a list of neuron models:

.. code-block:: yaml

    # neuron_catalogue.yml

    D1_spiny:
        neuron_type:  aeif
        C_m :         40.
        G_L :         200.0

        E_l:        -70.0
        V_reset:    -55.0
        V_peak:     0.0
        tau_refrac: 0.0

        # Exp pars
        delta_T:  2
        V_th:     -40

        # Adapting pars
        ada_a:      0.0
        ada_b:      5.0
        ada_tau_w:  100.0

        # Syn pars
        tau_ex:   10.
        tau_in:   5.5
        E_ex:     0.
        E_in:     -65.

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
            connectivity: 0.1
            weight:       0.5
            weight_delta: 0.05
            delay:        1.1
            delay_delta:  0.1
            type:         exc

      - name: STR1->GPi
        efferent: STR1
        afferent: GPi
        features:
            connectivity: 0.1
            weight:       0.03
            weight_delta: 0.001
            type:         inh

The parameters of a projection between an efferent population of size ``N`` and an afferent population of size ``M`` are (see `here <https://github.com/djanloo/quilt/issues/2>`_):

  - ``connectivity``: fraction of links over total (``N`` * ``M``). 
    It's the probability that neuron ``i`` of the efferent population will be linked to neuron ``j`` of the afferent population.
  - ``delay``: central value of delay (lognorm distributed)
  - ``delay_delta``: standard deviation of delay (lognorm distributed)
  - ``weight``: central value of weight (lognorm distributed)
  - ``weight_delta``: standard deviation of weight (lognorm distributed)
  - ``type``: ``inh`` or ``exc``


To build a spiking network:

.. code-block:: python

    from quilt.builder import SpikingNetwork
    spikenet = SpikingNetwork.from_yaml("network_model.yml", catalogue)

Oscillator networks
+++++++++++++++++++

.. warning::

  This is under construction

I/O and running
+++++++++++++++

.. code-block:: python

    # Adds a 25 pA current from t=10ms to t=20ms
    spikenet.populations["STR1"].add_injector(25.0, 10, 20)

    # Adds an excitatory poisson injector with rate 500 Hz and weight 0.1
    sn.populations["GPi"].add_poisson_spike_injector(500, 0.1)

    # Saves spikes
    spikenet.populations["STR1"].monitorize_spikes()

    # Saves neurons' states
    spikenet.populations["STR1"].monitorize_states()

    # Runs for 10 ms
    spikenet.run(dt=0.1, time=10)




in pure C++
-----------

.. warning::

  This is under construction
