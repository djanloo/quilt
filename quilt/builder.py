"""A module for building objects from configuration files.
"""
import os
from time import time
import copy

import yaml
from rich import print
from rich.progress import track

import quilt.interface.spiking as spiking
import quilt.interface.base_objects as base_objects

class SpikingNetwork:

    def __init__(self):
        self.is_built = False

        self.spike_monitored_pops = set()
        self.state_monitored_pops = set()

    def monitorize_spikes(self, populations=None):
        if populations is None:
            populations = self.populations.keys()
        
        # Keeps trace in case of rebuild
        self.spike_monitored_pops = self.spike_monitored_pops.union(populations)

        for pop in populations:
            self.populations[pop].monitorize_spikes()
    
    def monitorize_states(self, populations=None):
        if populations is None:
            populations = self.populations.keys()

        # Keeps trace in case of rebuild
        self.state_monitored_pops = self.state_monitored_pops.union(populations)

        for pop in populations:
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
    def from_yaml(cls, features_file, neuron_catalogue):
        net = cls()
        net._interface = None#spiking.SpikingNetwork("05535")

        net.features_file = features_file
        net.neuron_catalogue = copy.deepcopy(neuron_catalogue)
        
        if not os.path.exists(net.features_file):
            raise FileNotFoundError(f"YAML file '{net.features_file}' not found")
        
        with open(net.features_file, "r") as f:
            net.features_dict = yaml.safe_load(f)

        return net
    

    def build(self, progress_bar=None):

        # Destroys and rebuilds C++ network
        if self._interface is not None:
            del self._interface
        self._interface = spiking.SpikingNetwork("05535")

        # Adds back monitors
        self.monitorize_spikes(populations=self.spike_monitored_pops)
        self.monitorize_states(populations=self.state_monitored_pops)

        if progress_bar is None:
            if spiking.VERBOSITY == 1:
                progress_bar = True
            else:
                progress_bar = False

        self.populations = dict()
        
        # Builds populations
        if "populations" in self.features_dict and self.features_dict['populations'] is not None:
            for pop in self.features_dict['populations']:
                features = self.features_dict['populations'][pop]
                paramap = self.neuron_catalogue[features['neuron_model']]
                try:
                    self.populations[pop] = spiking.Population( features['size'], paramap, self._interface )
                except IndexError as e:
                    message = f"While building population {pop} an error was raised:\n\t"
                    message += str(e)
                    raise IndexError(message)
            
        # Builds projections
        if "projections" in self.features_dict and self.features_dict['projections'] is not None:
            if progress_bar:
                iter = track(self.features_dict['projections'], description="Building connections..")
            else:
                iter = self.features_dict['projections']
            for proj in iter:
                features = self.features_dict['projections'][proj]

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
        if "devices" in self.features_dict and self.features_dict['devices'] is not None:

            for device_name in self.features_dict['devices']:
                try:
                    hierarchical_level, target, description = device_name.split("_")
                except ValueError as e:
                    raise ValueError(f"Error while building device from yaml (format error?): {e}")
                device_features = self.features_dict['devices'][device_name]
                print(f"building device {device_name}")
                if hierarchical_level == "pop":
                    match device_features['type']:
                        case "poisson_spike_source":
                            t_min, t_max = None, None
                            try:
                                t_min = device_features['t_min']
                            except KeyError:
                                pass
                            try:
                                t_max = device_features['t_max']
                            except KeyError:
                                pass
                            self.populations[target].add_poisson_spike_injector(device_features['rate'], device_features['weight'], t_min=t_min, t_max=t_max)
                        case _:
                            raise NotImplementedError(f"Device of type '{device_features['type']}' is not implemented")
                else:
                    raise NotImplementedError(f"Device at hierarchical level '{hierarchical_level}' is not implemented")
                        
        else:
            print("No devices found")
        self.is_built = True

class ParametricSpikingNetwork(SpikingNetwork):

    @classmethod
    def from_yaml(cls, features_file, susceptibility_file, neuron_catalogue):
        net = super().from_yaml(features_file, neuron_catalogue)
        net.susceptibility_file = susceptibility_file

        if not os.path.exists(net.susceptibility_file):
            raise FileNotFoundError(f"YAML file '{net.susceptibility_file}' not found")
        
        with open(net.susceptibility_file, "r") as f:
            net.susceptibility_dict = yaml.safe_load(f)
        
        try:
            net.susceptibility_dict['parameters']
        except KeyError as e:
            raise KeyError("Susceptibility file must have a 'parameters' field")
        
        try:
            net.susceptibility_dict['parametric']
        except KeyError as e:
            raise KeyError("Susceptibility file must have a 'parametric' field")
    
        return net

    def set_parameters(self, **params):
        # Signals to the build method that this must be rebuilt
        self.is_built = False

        self.params_value = dict()
        self.params_range = dict()
        self.params_shift = dict()

        # Initializes all possible parameters to zero
        for param_name in self.susceptibility_dict['parameters']:

            # Initilaizes to 'shift' value to have zero driving force
            self.params_value[param_name] = self.susceptibility_dict['parameters'][param_name]['shift']
            self.params_shift[param_name] = self.susceptibility_dict['parameters'][param_name]['shift']
            self.params_range[param_name] = [self.susceptibility_dict['parameters'][param_name]['min'],
                                            self.susceptibility_dict['parameters'][param_name]['max']]

            # print(f"initialized parameter {param_name} with value {self.params_value[param_name]} "+
                #   f"and range {self.params_range[param_name]}")

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
        
        print(f"Building parametric network with params {self.params_value}")

        for param in self.params_value:
            for object in self.susceptibility_dict['parametric'][param]:
                # print(f"Parametrizing {object}")

                attribute = object['attribute']
                chi = object['susceptibility']
                parametric_relative_delta = chi * (self.params_value[param] - self.params_shift[param])

                if "population" in object:
                    parametric_populations = object['population'].split(',')
                    parametric_populations = [pop.strip() for pop in parametric_populations]

                    for pop in parametric_populations:

                        if pop not in self.features_dict['populations']:
                            raise KeyError(f"Population {pop} not found in network")
                        
                        is_pop_attr = False
                        is_neuron_attr = False

                        # Check if is a direct population attribute (size)
                        # I know it's uselessly too general, it's just in case I have add some pop attributes
                        try:
                            base_value = self.features_dict['populations'][pop][attribute]
                            self.features_dict['populations'][pop][attribute] = base_value * ( 1 + parametric_relative_delta)
                            is_pop_attr = True
                        except KeyError as error:
                            is_pop_attr = False

                        # Check if is a neuron attribute
                        try:
                            neuron_model = self.features_dict['populations'][pop]['neuron_model']
                            base_value = self.neuron_catalogue.neurons_dict[neuron_model][attribute]
                            self.neuron_catalogue.update(neuron_model, attribute, base_value * ( 1 + parametric_relative_delta))
                            is_neuron_attr = True
                        except KeyError as e:
                            is_neuron_attr = False
                        
                        if not is_neuron_attr and not is_pop_attr:
                            raise KeyError(f"Paremetric attribute '{attribute}' was specified on population '{pop}' but was not found neither in the population nor in the neuron model.")

                elif "projection" in object:
                    parametric_projections = object['projection'].split(",")
                    parametric_projections = [proj.strip() for proj in parametric_projections]

                    for proj in parametric_projections:

                        if proj not in self.features_dict['projections'].keys():
                            raise KeyError(f"Projection '{proj}' was not found in {list(self.features_dict['projections'].keys())}")

                        try:
                            base_value = self.features_dict['projections'][proj][attribute]
                            self.features_dict['projections'][proj][attribute] = base_value * ( 1 + parametric_relative_delta)

                        except KeyError as e:
                            raise KeyError(f"Paremetric attribute '{attribute}' was specified on projection '{proj}' but was not found.")
                
                else:
                    raise KeyError(f"Parametric object {object} was not found in the network")

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
            catalogue.paramaps[neuron_name] = base_objects.ParaMap(catalogue.neurons_dict[neuron_name])
            catalogue.neuron_names += [neuron_name]
        
        return catalogue
    
    def update(self, neuron_model, attribute, value):
        if attribute in self.neurons_dict[neuron_model].keys():
            self.neurons_dict[neuron_model][attribute] = value
            self.paramaps[neuron_model] = base_objects.ParaMap(self.neurons_dict[neuron_model])
        else:
            print(f"neuron model '{neuron_model}' raised error")
            raise KeyError(f"Neuron model {neuron_model} has no attribute {attribute}")

    
    def __getitem__(self, neuron):
        try:
            paramap = self.paramaps[neuron]
        except KeyError:
            raise KeyError(f"Neural model '{neuron}' does not exist in this catalogue")

        return paramap
