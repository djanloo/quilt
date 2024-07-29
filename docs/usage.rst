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

    D1_neuron:
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
    
    populations:
      D1:
        size: 6000
        neuron_model: D1_neuron
      D2:
        size: 6000
        neuron_model: D2_neuron

    # Note: projections must be in the A->B format
    projections:
      D1->D1:
        fan_in:     364
        delay:      1.7
        weight:     0.12
        type:       inh

      D1->D2:
        fan_in:     84
        delay:      1.7
        weight:     0.30
        type:       inh

      D2->D1:
        fan_in:     392
        delay:      1.7
        weight:     0.36
        type:       inh

The parameters of a projection between an efferent population of size ``N`` and an afferent population of size ``M`` are (see `here <https://github.com/djanloo/quilt/issues/2>`_):

  - ``fan_in``: average incoming synapses of the afferent neuron 
  - ``delay``: central value of delay (lognorm distributed)
  - ``delay_delta``: standard deviation of delay (lognorm distributed)
  - ``weight``: central value of weight (lognorm distributed)
  - ``weight_delta``: standard deviation of weight (lognorm distributed)
  - ``type``: ``inh`` or ``exc``


To build a spiking network:

.. code-block:: python

    from quilt.builder import SpikingNetwork
    spikenet = SpikingNetwork.from_yaml("network_model.yml", "neuron_catalogue.yml")

Oscillator networks
^^^^^^^^^^^^^^^^^^^

To build an oscillator (cortical) network a large-scale connectivity of the network is needed. 
The most common format of large-scale connectivities it the format of The Virtual Brain. A TVB connectome is made of

  - a dictionary of the geometrical centers of the regions
  - a matrix of weights
  - a matrix of delays

often grrouped in a zip file.

Given that, a homogeneous network can be constructed with

.. code-block:: python
  
  oscnet = OscillatorNetwork.homogeneous_from_TVB("/connectivity_desikan.zip", 
                                                {'oscillator_type':'jansen-rit',
                                                'U':0.12}, 
                                                global_weight=5.0, 
                                                conduction_speed=1.0)

The dictionary provided to the function is used to set the parameters of the oscillators.

.. warning::

  Check the spelling for the name of parameters in the documentation of each oscillator since no errors are raised for mispelled parameters.


Multiscale networks
^^^^^^^^^^^^^^^^^^^

Multiscale networks require an instance of :py:class:`quilt.builder.SpikingNetwork` and :py:class:`quilt.builder.OscillatorNetwork`.
Furthermore a multiscale connectome must be given, but this will be explained in the next section.

.. code-block:: python

  multinet = MultiscaleNetwork(spikenet, oscnet, "./transducers.yaml")
  multinet.set_multiscale_projections(file="putamen_weights.yaml", 
                                    T2O_coupling=0.2, 
                                    O2T_coupling=10.0)

Multiscale connectome
^^^^^^^^^^^^^^^^^^^^^

Transducers (supersynapses) target only one population each.
Their parameters are set in a supersynapse config file:

.. code-block:: yaml

  # transducers.yaml

  transducers:
    - name: D1_td
      population: D1
      initialization_rate: 500
      weight: 0.3
      weight_delta: 0.05
      generation_window: 5

    - name: D2_td
      population: D2
      initialization_rate: 500
      weight: 0.3
      weight_delta: 0.05
      generation_window: 5


Also, the connectivity of the supersynapses to each cortical oscillator must be set in a configuration file like:

.. code-block:: yaml

  # putamen_weights.yaml
  D1_td:
    incoming:
      r_superiorfrontal: 
        weight: 13.13
        delay: 10
      l_superiorfrontal: 
        weight: 13.13
        delay: 10
     # ...

  SNR_td:
    outgoing:
      r_superiorfrontal: 
        weight: 13.13
        delay: 10
      l_superiorfrontal: 
        weight: 13.13
        delay: 10
      # ...


I/O and running
^^^^^^^^^^^^^^^

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
