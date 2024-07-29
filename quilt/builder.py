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
import quilt.interface.multiscale as multi

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

    @property
    def n_populations(self):
        return len(self.populations.keys())
    
    def pop_id(self, pop_name):
        return {pop:i for pop, i in zip(self.populations, range(self.n_populations))}[pop_name]

class ParametricSpikingNetwork(SpikingNetwork):

    @classmethod
    def from_dict(cls, susceptibility_dict, network=None, network_file=None, neuron_file=None):

        if network is not None:
            net = network
        elif network_file is not None and neuron_file is not None:
            net = super().from_yaml(network_file, neuron_file)
    
        # Backups for parametrization
        net.original_features = copy.deepcopy(net.features)
        net.original_neuron_catalogue = copy.deepcopy(net.neuron_catalogue)

        net.susceptibility_dict = susceptibility_dict
        try:
            net.susceptibility_dict['parameters']
        except KeyError as e:
            raise KeyError(f"Susceptibility dict must have a 'parameters' field")
        
        try:
            net.susceptibility_dict['parametric']
        except KeyError as e:
            raise KeyError(f"Susceptibility dict must have a 'parametric' field")
        
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

    @classmethod
    def from_yaml(cls, network_file, neuron_file, susceptibility_files):

        # In case is a single file
        if isinstance(susceptibility_files, str):
            susceptibility_files = [susceptibility_files]

        # Loads parameters
        susceptibility_dict = dict(parameters=dict(), parametric=dict())
        for susceptibility_file in susceptibility_files:
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
            susceptibility_dict['parameters'].update(chi_dict['parameters'])
            susceptibility_dict['parametric'].update(chi_dict['parametric'])
        
        return ParametricSpikingNetwork.from_dict(susceptibility_dict, network_file=network_file, neuron_file=neuron_file)

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

        # Conversion to float of each parameter except neuron type
        for neuron_name in catalogue.neurons_dict.keys():
            for feature in catalogue.neurons_dict[neuron_name]:
                try:
                    catalogue.neurons_dict[neuron_name][feature] = float(catalogue.neurons_dict[neuron_name][feature])
                except ValueError:
                    # print(f"Could not convert to float: {feature}->{catalogue.neurons_dict[neuron_name][feature]}")
                    pass

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

        self.n_oscillators = None
        self.connectivity = None
        self.oscillators = dict()

        # Here is stored the information about network structure
        # without having to build it
        self.features = dict()

        # This is the parameter dict
        # used in case the network is homogeneous
        self.homogeneous_dict = None

        # This is the list of dicts
        # used in case the nerwork is inhomogeneous
        self.inhomogeneous_list_of_dicts = None

        self.oscillators = dict()
    
    def initialize(self, states, dt=0.1):
        self._interface.initialize(states, dt=dt)

    def run(self, dt=0.2, time=1):
        if not self.is_built:
            self.build()
        self._interface.run(time=time)
    
    def build(self):

        if self.homogeneous_dict is not None:
            self._interface = oscill.OscillatorNetwork.homogeneous(self.n_oscillators,  base.ParaMap(self.homogeneous_dict))
        
        # Creates the dictionary of the oscillators
        self.oscillators = {n:o for n,o in zip(self.features["oscillators"], self._interface.oscillators)}
        
        # Makes the connections
        projection = base.Projection(   
                                    self.features['connectivity']['weights'], 
                                    self.features['connectivity']['delays']
                                    )
        # ACHTUNG !!: for now the links take a blank paramap 
        self._interface.build_connections(projection, base.ParaMap({}))

        self.is_built = True
    
    @classmethod
    def homogeneous_from_TVB(cls, connectivity_file, oscillator_parameters, global_weight=1.0, conduction_speed=1.0):

        net = cls()
        net.oscillators = dict()
        net.features['oscillators'] = dict()
        net.features['connectivity'] = dict()
        net.features['centers'] = dict()
    
        with zipfile.ZipFile(connectivity_file, 'r') as zip_ref:

            # Load oscillator names
            if "centres.txt" in zip_ref.namelist():
                centres = zip_ref.read("centres.txt")
                for line in centres.decode('utf-8').splitlines():
                    name, x, y, z = line.split()
                    net.oscillators[name] = None # Adds them with None so they can be listed
                    net.features['oscillators'][name] = oscillator_parameters # Stores duplicates of the parameter for a-posteriori inhomogeneity
                    net.features['centers'][name] = np.array([float(v) for v in [x,y,z]])
            else:
                print(f"centres.txt not in connectivity.")
            
            net.n_oscillators = len(net.features['oscillators'])

            # Load tract lengths
            if "tract_lengths.txt" in zip_ref.namelist():
                tracts = zip_ref.read("tract_lengths.txt").decode('utf-8')
                net.features['connectivity']['delays'] = 1/conduction_speed * np.loadtxt(tracts.splitlines())
            else:
                print(f"tract_lengths.txt not in connectivity.")
            
            # Load weights
            if "weights.txt" in zip_ref.namelist():
                tracts = zip_ref.read("weights.txt").decode('utf-8')
                net.features['connectivity']['weights'] = global_weight * np.loadtxt(tracts.splitlines())
            else:
                print(f"weights.txt not in connectivity.")
        
        # Checks for vanishing delays
        real_connections = (net.features['connectivity']['weights'] > 0)
        vanishing_delays = (net.features['connectivity']['delays'][real_connections] == 0)

        if np.sum(vanishing_delays) > 1:
            # import matplotlib.pyplot as plt
            real_delays = net.features['connectivity']['delays']
            real_delays[~real_connections] = np.nan
            # print(real_delays)
            # plt.matshow(real_delays)
            # plt.colorbar()
            # plt.show()

            warnings.warn("While loading TVB data: a delay of 0 ms is not supported"+f" ({int(np.sum(vanishing_delays))} over {np.sum(real_connections)} are vanishing)")

        net.n_oscillators = len(net.features['oscillators'])
        net.homogeneous_dict = oscillator_parameters

        # Converts the connectivity to float32
        net.features['connectivity']['weights'] = net.features['connectivity']['weights'].astype(np.float32)
        net.features['connectivity']['delays'] = net.features['connectivity']['delays'].astype(np.float32)

        return net
    
    def reg_id(self, reg_name):
        return {reg:i for reg, i in zip(self.oscillators, range(self.n_oscillators))}[reg_name]

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
            time_series[k] = network.oscillators[oscillator_name].eeg # Uses the eeg method of oscillator that gives the right VOI

        for j in range(self.n_electrodes):
            for k in range(self.n_regions):
                signal[j] += self.weights[j, k] * time_series[k] 
            if filter_signal:
                signal[j] = sosfiltfilt(sos, signal[j])
        return signal


class MultiscaleNetwork:

    def __init__(self, 
                 spiking_network : SpikingNetwork,
                 oscillator_network: OscillatorNetwork,
                 transducers: str | list
                 ):
        self.spiking_network = spiking_network
        self.oscillator_network = oscillator_network
        self.features = dict()

        # Adds transducers
        transducers_list = []
        self.features['transducers'] = dict()
        if type(transducers) is str:

            if not os.path.exists(transducers):
                raise FileNotFoundError("YAML file not found")
            
            with open(transducers, "r") as f:
                transducers_list = yaml.safe_load(f)['transducers']

        elif type(transducers) is list:
            transducers_list = transducers

        for td in transducers_list:
            self.features['transducers'][td['name']] = {key:val for key, val in td.items() if key != 'name'}

        self.is_built = False
    
    @property
    def n_transducers(self):
        return len(self.features['transducers'].keys())
    
    @property
    def n_oscillators(self):
        return self.oscillator_network.n_oscillators
    
    @property
    def n_populations(self):
        return self.spiking_network.n_populations

    @property
    def transducer_histories(self):
        return self._interface.transducer_histories
    
    def set_multiscale_projections(self, file=None, 
                                    T2O_coupling=None, O2T_coupling=None, 
                                    T2O=None, O2T=None):

        if file is not None:
            T2O, O2T = self._get_mproj_from_yaml(file, T2O_coupling=T2O_coupling, O2T_coupling=O2T_coupling)
            self.features["O2T_projection"] = O2T
            self.features["T2O_projection"] = T2O
        elif T2O is not None and O2T is not None:
            self.features["O2T_projection"] = O2T
            self.features["T2O_projection"] = T2O
        else:
            raise ValueError("File or projection must be specified")
    
    def _get_mproj_from_yaml(self, file, T2O_coupling=None, O2T_coupling=None):
        with open(file, 'r') as f:
            proj_dict = yaml.safe_load(f)

        # Checks that transducers match
        for td in self.features['transducers']:
            if td not in proj_dict.keys():
                raise ValueError(f"Transducer {td} is missing from connectivity file")

        # Check that all population exists
        nonexistent_td = set()
        nonexistent_regions = set()
        for td in proj_dict:
            if td not in self.features['transducers']:
                nonexistent_td.add(td)

            td_options = list(proj_dict[td].keys())
            valid_options = ['outgoing', 'incoming']
            unvalid_options = [opt for opt in td_options if opt not in valid_options]

            if unvalid_options:
                raise ValueError(f"Multiscale connectivity can only specify 'incoming' and 'outgoing' properties. Unvalid options: {unvalid_options}")
            
            for direction in ['incoming', 'outgoing']:
                try:
                    # Check that all region exist
                    for region in proj_dict[td][direction]:
                        if region not in self.oscillator_network.oscillators.keys():
                            nonexistent_regions.add(region)
                except KeyError:
                    pass

        if nonexistent_regions:
            raise KeyError(f"Regions {nonexistent_regions} were specified by multiscale connectivity but are not present in oscillator network")
        if nonexistent_td:
            raise KeyError(f"Transducers {nonexistent_td} were specified by multiscale connectivity but are not present in transducers")
    
        # Builds the matrices: T2O
        T2O_w = np.zeros((len(proj_dict), self.n_oscillators))
        T2O_d = np.zeros((len(proj_dict), self.n_oscillators))

        for td in proj_dict:
            try:
                reg_dict = proj_dict[td]['outgoing']
                for reg in reg_dict:
                    T2O_w[self.transducer_id(td), self.oscillator_network.reg_id(reg)] = reg_dict[reg]['weight']
                    T2O_d[self.transducer_id(td), self.oscillator_network.reg_id(reg)] = reg_dict[reg]['delay']
            except KeyError:
                pass

        O2T_w = np.zeros((self.n_oscillators, len(proj_dict)))
        O2T_d = np.zeros((self.n_oscillators, len(proj_dict)))
        
        # Builds the matrices: O2T
        for td in proj_dict:
            try:
                reg_dict = proj_dict[td]['incoming']
                for reg in reg_dict:
                    O2T_w[self.oscillator_network.reg_id(reg), self.transducer_id(td)] = reg_dict[reg]['weight']
                    O2T_d[self.oscillator_network.reg_id(reg), self.transducer_id(td),] = reg_dict[reg]['delay']

            except KeyError:
                pass
        
        if T2O_coupling is not None:
            T2O_w *= T2O_coupling
        
        if O2T_coupling is not None:
            O2T_w *= O2T_coupling

        return base.Projection(T2O_w.astype(np.float32), T2O_d.astype(np.float32)), base.Projection(O2T_w.astype(np.float32), O2T_d.astype(np.float32))
            
    def set_evolution_contextes(self, dt_short=0.1, dt_long=1.0):
        if not self.is_built:
            raise RuntimeError("Network must be built before setting the evolution contextes")
        self._interface.set_evolution_contextes(dt_short, dt_long)

    def initialize(self, states):
        if not self.is_built:
            self.build()
        self._interface.initialize(states)
    
    def run(self, time):
        if not self.is_built:
            self.build()
            raise RuntimeError("Network must be initialized first")
        self._interface.run(time=time)

    def build(self):
        self.spiking_network.build()
        self.oscillator_network.build()

        # Builds C++ objects
        self._interface = multi.MultiscaleNetwork(self.spiking_network._interface, self.oscillator_network._interface)

        for td in self.features['transducers']:
            population = self.spiking_network.populations[self.features['transducers'][td]['population']]
            params = base.ParaMap(self.features['transducers'][td])

            self._interface.add_transducer(population, params)

        self._interface.build_multiscale_projections(self.features['T2O_projection'], self.features['O2T_projection'])
        self.is_built = True

    def transducer_id(self, td_name):
        return {td: i for i, td in enumerate(self.features['transducers'])}[td_name]