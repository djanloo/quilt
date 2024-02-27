"""A module for building objects from configuration files.
"""
import os
from time import time
import copy
import warnings
import zipfile

import yaml
import numpy as np
from rich import print
from rich.progress import track
from scipy.signal import butter, sosfiltfilt

import quilt.interface.base as base
import quilt.interface.spiking as spiking
import quilt.interface.oscill as oscill

class SpikingNetwork:

    def __init__(self):
        self.is_built = False
        self.populations = None
        self._interface = None

        self.spike_monitored_pops = set()
        self.state_monitored_pops = set()

        # Here are stored the information about the structure
        # without a built network.
        # Features is a dict having possible fields ['populations', 'projections', 'devices']
        self.features = dict() 

    def monitorize_spikes(self, populations=None):
        if self.is_built:
            warnings.warn("Adding monitors after building the network will trigger another rebuild")
            self.is_built = False

        # Monitors are built after populations
        # This is needed for rebuilding
        if populations is None:
            populations = self.features['populations'].keys()
        elif isinstance(populations, str):
            populations = [populations]
        
        self.spike_monitored_pops = self.spike_monitored_pops.union(populations)

    def monitorize_states(self, populations=None):
        if self.is_built:
            warnings.warn("Adding monitors after building the network will trigger another rebuild")
            self.is_built = False

        # Monitors are built after populations
        # This is needed for rebuilding
        if populations is None:
            populations = self.features['populations'].keys()
        elif isinstance(populations, str):
            populations = [populations]
    
        self.state_monitored_pops = self.state_monitored_pops.union(populations)

    def _build_monitors(self):
        for pop in self.spike_monitored_pops:
            self.populations[pop].monitorize_spikes()
        
        for pop in self.state_monitored_pops:
            self.populations[pop].monitorize_states()

    
    def run(self, dt=0.1, time=1):

        if not self.is_built:
            self.build()
        self._interface.run(dt, time)

    @property
    def interface(self):
        raise AttributeError("SpikingNetwork interface is not meant to be accessed from here")
    
    @interface.setter
    def interface(self, interface):
        self._interface = interface

    @classmethod
    def from_yaml(cls, features_file, neuron_file):
        net = cls()
        net._interface = None#spiking.SpikingNetwork("05535")

        net.features_file = features_file
        net.neuron_catalogue = NeuronCatalogue.from_yaml(neuron_file)
        
        if not os.path.exists(net.features_file):
            raise FileNotFoundError(f"YAML file '{net.features_file}' not found")
        
        with open(net.features_file, "r") as f:
            net.features = yaml.safe_load(f)

        # Converts to connectivity in case fan-in is specified
        try:
            for proj in net.features['projections']:
                features = net.features['projections'][proj]

                efferent, _ = proj.split("->")
                efferent = efferent.strip()
                
                if "fan_in" in features.keys():
                    if features["fan_in"] < 0:
                        raise ValueError("in projection {proj}: fan-in must be greater than zero")
                    elif features["fan_in"] < 1:
                        raise ValueError("in projection {proj}: fan-in must be greater than one")
                    
                    if "connectivity" in features.keys():
                        warnings.warn(f"While building projection {proj}, fan-in overrided connectivity")
                    features['connectivity'] = features['fan_in']/net.features['populations'][efferent]['size']
                    del features['fan_in']
        except KeyError:
            pass

        return net
    
    def refresh_all(self):
        # Destroys and rebuilds C++ network
        if self._interface is not None:
            del self._interface
        
        if self.populations is not None:
            for pop in self.populations:
                del pop
    

    def build(self, progress_bar=False):

        self.refresh_all()
        self._interface = spiking.SpikingNetwork("05535")

        if progress_bar is None:
            if spiking.VERBOSITY == 1:
                progress_bar = True
            else:
                progress_bar = False

        self.populations = dict()
        
        # Builds populations
        if "populations" in self.features and self.features['populations'] is not None:
            for pop in self.features['populations']:
                features = self.features['populations'][pop]
                paramap = self.neuron_catalogue[features['neuron_model']]
                try:
                    self.populations[pop] = spiking.Population( features['size'], paramap, self._interface )
                except IndexError as e:
                    message = f"While building population {pop} an error was raised:\n\t"
                    message += str(e)
                    raise IndexError(message)
            
        # Builds projections
        if "projections" in self.features and self.features['projections'] is not None:
            if progress_bar:
                iter = track(self.features['projections'], description="Building connections..")
            else:
                iter = self.features['projections']
            for proj in iter:
                features = self.features['projections'][proj]

                efferent, afferent = proj.split("->")
                efferent, afferent = efferent.strip(), afferent.strip()
                
                if efferent not in self.populations.keys():
                    raise KeyError(f"In projection {efferent} -> {afferent}: <{efferent}> was not defined")
                if afferent not in self.populations.keys():
                    raise KeyError(f"In projection {efferent} -> {afferent}: <{afferent}> was not defined")

                try:
                    # Builds the projector
                    projector = spiking.SparseProjector(features, dist_type="lognorm")
                except ValueError as e:
                    raise ValueError(f"Some value was wrong during projection {efferent}->{afferent}:\n{e}")
                
                efferent = self.populations[efferent]
                afferent = self.populations[afferent]
                efferent.project(projector.get_projection(efferent, afferent), afferent)

        # Builds external devices
        if "devices" in self.features and self.features['devices'] is not None:

            for device_name in self.features['devices']:
                try:
                    hierarchical_level, target, description = device_name.split("_")
                except ValueError as e:
                    raise ValueError(f"Error while building device from yaml (format error?): {e}")
                device_features = self.features['devices'][device_name]

                if hierarchical_level == "pop":
                    match device_features['type']:
                        case "poisson_spike_source":

                            t_min = device_features.get('t_min', None)
                            t_max = device_features.get('t_max', None)
                            weight_delta = device_features.get('weight_delta', 0)
                            self.populations[target].add_poisson_spike_injector(device_features['rate'], device_features['weight'], weight_delta, t_min=t_min, t_max=t_max)
                        case _:
                            raise NotImplementedError(f"Device of type '{device_features['type']}' is not implemented")
                else:
                    raise NotImplementedError(f"Device at hierarchical level '{hierarchical_level}' is not implemented")
                        
        else:
            print("No devices found")


        # Adds back the monitors
        self._build_monitors()
        self.is_built = True

class ParametricSpikingNetwork(SpikingNetwork):

    @classmethod
    def from_yaml(cls, network_file,  
                        neuron_file,
                        susceptibility_files):
        net = super().from_yaml(network_file, neuron_file)

        # Backups for parametrization
        net.original_features = copy.deepcopy(net.features)
        net.original_neuron_catalogue = copy.deepcopy(net.neuron_catalogue)

        net.susceptibility_files = susceptibility_files
        # In case is a single file
        if isinstance(net.susceptibility_files, str):
            net.susceptibility_files = [net.susceptibility_files]

        # Loads parameters
        net.susceptibility_dict = dict(parameters=dict(), parametric=dict())
        for susceptibility_file in net.susceptibility_files:
            if not os.path.exists(susceptibility_file):
                raise FileNotFoundError(f"YAML file '{susceptibility_file}' not found")
            
            with open(susceptibility_file, "r") as f:
                chi_dict = yaml.safe_load(f)            
            try:
                chi_dict['parameters']
            except KeyError as e:
                raise KeyError(f"Susceptibility file {susceptibility_file} must have a 'parameters' field")
            
            try:
                chi_dict['parametric']
            except KeyError as e:
                raise KeyError(f"Susceptibility file {susceptibility_file} must have a 'parametric' field")

            # Adds to parameters
            net.susceptibility_dict['parameters'].update(chi_dict['parameters'])
            net.susceptibility_dict['parametric'].update(chi_dict['parametric'])

        net.params_value = dict()
        net.params_range = dict()
        net.params_shift = dict()

        # Initializes all possible parameters to default (shift) value so the have no 'driving force'
        for param_name in net.susceptibility_dict['parameters']:

            net.params_value[param_name] = net.susceptibility_dict['parameters'][param_name].get('shift',0)
            net.params_shift[param_name] = net.susceptibility_dict['parameters'][param_name].get('shift',0)
            net.params_range[param_name] = [net.susceptibility_dict['parameters'][param_name].get('min',0),
                                            net.susceptibility_dict['parameters'][param_name].get('max',1)]
        return net

    def set_parameters(self, **params):
        # Signals to the run method that this must be rebuilt
        self.is_built = False

        # Reset features
        self.features = copy.deepcopy(self.original_features)
        self.neuron_catalogue = copy.deepcopy(self.original_neuron_catalogue)

        # Checks that specified params are contained in possible params
        for param in params.keys():
            if param not in self.params_value.keys():
                raise ValueError(f"Parameter {param} not in available parameters {list(self.params_value.keys())}")
            
        # Assigns the specified parameters of the network
        self.params_value.update(params)
        del params # Now use only self.params_value

        # Check that parameter is in range
        for param in self.params_value:
            if self.params_value[param] < self.params_range[param][0] or self.params_value[param] > self.params_range[param][1]:
                raise ValueError(f"Value {self.params_value[param]} for parameter '{param}' is not in range {self.params_range[param]}")
        

        for param in self.params_value:
            for parametric_object in self.susceptibility_dict['parametric'][param]:

                attribute = parametric_object['attribute']


                # Choses action: if susceptibility is not specified, use action 'set'
                # Not the most elegant way of doing this
                action = "set"
                if "susceptibility" in parametric_object:
                    action = "multiply"

                if "population" in parametric_object:

                    if parametric_object['population'] == "ALL":
                        parametric_populations = self.features['populations'].keys()
                    else:
                        parametric_populations = parametric_object['population'].split(",")
                        parametric_populations = [pop.strip() for pop in parametric_populations]

                    for pop in parametric_populations:

                        if pop not in self.features['populations']:
                            raise KeyError(f"Population {pop} not found in network")
                        
                        is_pop_attr = False
                        is_neuron_attr = False

                        # Check if is a direct population attribute (size)
                        # I know it's uselessly too general, it's just in case I have add some pop attributes
                        try:
                            base_value = self.features['populations'][pop][attribute]
                            match action:
                                case 'set':
                                    self.features['populations'][pop][attribute] = self.params_value[param]
                                case 'multiply':
                                    self.features['populations'][pop][attribute] = base_value * ( 1 + parametric_object['susceptibility'] * (self.params_value[param] - self.params_shift[param]))
                                    
                            is_pop_attr = True
                        except KeyError as error:
                            is_pop_attr = False

                        # Check if is a neuron attribute
                        try:
                            neuron_model = self.features['populations'][pop]['neuron_model']
                            base_value = self.neuron_catalogue.neurons_dict[neuron_model][attribute]

                            match action:
                                case 'set':
                                    self.neuron_catalogue.update(neuron_model, attribute, self.params_value[param])
                                case 'multiply':
                                    self.neuron_catalogue.update(neuron_model, attribute, base_value * ( 1 + parametric_object['susceptibility'] * (self.params_value[param] - self.params_shift[param])))
                            
                            is_neuron_attr = True
                        except KeyError as e:
                            is_neuron_attr = False

                        if not is_neuron_attr and not is_pop_attr:
                            raise KeyError(f"Paremetric attribute '{attribute}' was specified on population '{pop}' but was not found neither in the population nor in the neuron model.")

                elif "projection" in parametric_object:
                    if parametric_object['projection'] == "ALL":
                        parametric_projections = self.features['projections'].keys()
                    else:
                        parametric_projections = parametric_object['projection'].split(",")
                        parametric_projections = [proj.strip() for proj in parametric_projections]

                    for proj in parametric_projections:

                        if proj not in self.features['projections'].keys():
                            raise KeyError(f"Projection '{proj}' was not found in {list(self.features['projections'].keys())}")

                        try:
                            base_value = self.features['projections'][proj][attribute]
                            match action:
                                case 'set':
                                    self.features['projections'][proj][attribute] = self.params_value[param]
                                case 'multiply':
                                    self.features['projections'][proj][attribute] = base_value * ( 1 + parametric_object['susceptibility'] * (self.params_value[param] - self.params_shift[param]))

                        except KeyError as e:
                            raise KeyError(f"Parametric attribute '{attribute}' was specified on projection '{proj}' but was not found.")
                
                elif "device" in parametric_object:
                    parametric_devices = parametric_object['device'].split(",")
                    parametric_devices = [dev.strip() for dev in parametric_devices]

                    for dev in parametric_devices:

                        if dev not in self.features['devices'].keys():
                            raise KeyError(f"Device {dev} was not found in {list(self.features['devices'].keys())}")
                        try:
                            base_value = self.features['devices'][dev][attribute]
                            match action:
                                case 'set':
                                    self.features['devices'][dev][attribute] = self.params_value[param]
                                case 'multiply':
                                    self.features['devices'][dev][attribute] = base_value * ( 1 + parametric_object['susceptibility'] * (self.params_value[param] - self.params_shift[param]))

                        except KeyError as e:
                            raise KeyError(f"Parametric attribute '{attribute}' was specified on device '{dev}' but was not found.")
                
                else:
                    raise KeyError(f"Parametric object {parametric_object} was not found in the network")

class NeuronCatalogue:

    def __init__(self):
        pass

    @classmethod
    def from_yaml(cls, yaml_file):
        catalogue = cls()
        catalogue.yaml_file = yaml_file
        catalogue.paramaps = dict()
        catalogue.neuron_names = []
        
        if not os.path.exists(catalogue.yaml_file):
            raise FileNotFoundError("YAML file not found")
        
        with open(catalogue.yaml_file, "r") as f:
            catalogue.neurons_dict = yaml.safe_load(f)

        for neuron_name in catalogue.neurons_dict.keys():
            catalogue.paramaps[neuron_name] = base.ParaMap(catalogue.neurons_dict[neuron_name])
            catalogue.neuron_names += [neuron_name]
        
        return catalogue
    
    def update(self, neuron_model, attribute, value):
        if attribute in self.neurons_dict[neuron_model].keys():
            self.neurons_dict[neuron_model][attribute] = value
            self.paramaps[neuron_model] = base.ParaMap(self.neurons_dict[neuron_model])
        else:
            print(f"neuron model '{neuron_model}' raised error")
            raise KeyError(f"Neuron model {neuron_model} has no attribute {attribute}")

    
    def __getitem__(self, neuron):
        try:
            paramap = self.paramaps[neuron]
        except KeyError:
            raise KeyError(f"Neuron model '{neuron}' does not exist in this catalogue")
        return paramap

class OscillatorNetwork:

    def __init__(self):
        self.is_built = False
        self._interface = None
        self.connectivity = None
        self.oscillators = dict()

        # Here is stored the information about network structure
        # without having to build it
        self.features = dict()
    
    def init(self, states, dt=0.1):
        self._interface.init(states, dt=dt)

    def run(self, dt=0.2, time=1):
        if not self.is_built:
            self.build()
        self._interface.run(dt, time)
    
    def build(self):
        self._interface = oscill.OscillatorNetwork()
        self.oscillators = dict()

        for oscillator_name in self.features['oscillators']:
            osctype = self.features['oscillators'][oscillator_name]['oscillator_type']
            params = self.features['oscillators'][oscillator_name]
            self.oscillators[oscillator_name] = oscill.get_class[osctype](params, self._interface)
        
        try:
            weights = np.array(self.features['connectivity']['weights']).astype(np.float32)
            delays = np.array(self.features['connectivity']['delays']).astype(np.float32)
            proj = base.Projection(weights, delays)

            self._interface.build_connections(proj)
        except KeyError as e:
            raise KeyError(f"Missing parameter while building OscillatorNetwork connectivity: {e}")

        self.is_built = True
    
    @classmethod 
    def homogeneous(cls, oscillator_parameters, weights, delays, names=None):
        n_oscillators = len(weights)
        names = [f"osc_{i}" for i in range(n_oscillators)] if names is None else names
        if len(names) != n_oscillators:
            raise ValueError("List of oscillator names must have len() equal to n_oscillators")
        net = cls()

        net.features['oscillators'] = dict()
        for name in names:
            net.features['oscillators'][name] = oscillator_parameters

        net.features['connectivity'] = dict()
        net.features['connectivity']['weights'] = weights
        net.features['connectivity']['delays'] = delays

        return net
    
    @classmethod
    def homogeneous_from_TVB(cls, connectivity_file, oscillator_parameters, global_weight=1.0, conduction_speed=1.0):

        net = cls()
        net.features['oscillators'] = dict()
        net.features['connectivity'] = dict()
        net.features['centers'] = dict()
    
        with zipfile.ZipFile(connectivity_file, 'r') as zip_ref:

            # Load oscillator names
            if "centres.txt" in zip_ref.namelist():
                centres = zip_ref.read("centres.txt")
                for line in centres.decode('utf-8').splitlines():
                    name, x,y,z = line.split()
                    net.features['oscillators'][name] = oscillator_parameters
                    net.features['centers'][name] = np.array([float(v) for v in [x,y,z]])
            else:
                print(f"centres.txt not in connectivity.")
            
            net.n_oscillators = len(net.features['oscillators'])

            # Load tract lengths
            if "tract_lengths.txt" in zip_ref.namelist():
                tracts = zip_ref.read("tract_lengths.txt").decode('utf-8')
                net.features['connectivity']['delays'] = conduction_speed * np.loadtxt(tracts.splitlines())
            else:
                print(f"tract_lengths.txt not in connectivity.")
            
            # Load weights
            if "weights.txt" in zip_ref.namelist():
                tracts = zip_ref.read("weights.txt").decode('utf-8')
                net.features['connectivity']['weights'] = global_weight * np.loadtxt(tracts.splitlines())
            else:
                print(f"weights.txt not in connectivity.")
            

        return net

class EEGcap:
    def __init__(self, region_mapping_file, eeg_gain_file):
        self.regions = np.loadtxt(region_mapping_file)
        self.eeg_gain = np.load(eeg_gain_file)

        self.n_regions = len(np.unique(self.regions))
        self.n_electrodes = self.eeg_gain.shape[0]

        self.weights = np.zeros((self.n_electrodes, self.n_regions))

        for j in range(self.n_electrodes):
            for k in range(self.n_regions):
                node_is_in_region = (self.regions == k)
                self.weights[j, k] = np.sum(self.eeg_gain[j, node_is_in_region])
    
    def eeg(self, network,
            bandpass_edges=[0.5, 140], 
            sampling_frequency=1e3,
            filter_signal=True
            ):

        T = len(network.oscillators[list(network.oscillators.keys())[0]].history)
        signal = np.zeros((self.n_electrodes, T))
        sos = butter(5, bandpass_edges, 'bandpass', fs=sampling_frequency, output='sos')

        # Takes the timeseries once to avoid overhead due to data request
        time_series = np.zeros((self.n_regions, T))

        for k, oscillator_name in enumerate(network.oscillators):
            time_series[k] = network.oscillators[oscillator_name].history[:,0]

        for j in range(self.n_electrodes):
            for k in range(self.n_regions):
                signal[j] += self.weights[j, k] * time_series[k] 
            if filter_signal:
                signal[j] = sosfiltfilt(sos, signal[j])
        return signal


class MultiscaleNetwork:

    def __init__(self, networks, interscale_connectome):
        self.features = dict()
        for network in networks:
            if isinstance(network, SpikingNetwork):
                self.features
                pass
            elif isinstance(network, OscillatorNetwork):
                pass
            else:
                raise TypeError("Multiscale components must be a SpikingNetwork or an OscillatorNetwork")