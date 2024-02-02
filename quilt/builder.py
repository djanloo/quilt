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

        self.population_rescale_factor = 1
        self.connectivity_rescale_factor = 1
        self.weight_rescale_factor = 1
        self.delay_rescale_factor = 1

    def run(self, dt=0.1, time=1):

        if not self.is_built:
            self.build()
        self._interface.run(dt, time)

    def rescale_populations(self, population_rescale_factor):
        self.population_rescale = population_rescale_factor
    
    def rescale_connectivity(self, connectivity_rescale_factor):
        self.connectivity_rescale_factor = connectivity_rescale_factor
    
    def rescale_weights(self, weight_rescale_factor):
        self.weight_rescale_factor = weight_rescale_factor

    def rescale_delays(self, delay_rescale_factor):
        self.delay_rescale_factor = delay_rescale_factor

    @property
    def interface(self):
        raise AttributeError("SpikingNetwork interface is not meant to be accessed from here")
    
    @interface.setter
    def interface(self, interface):
        self._interface = interface

    @classmethod
    def from_yaml(cls, features_file, neuron_catalogue):
        net = cls()
        net._interface = spiking.SpikingNetwork("dummy")

        net.features_file = features_file
        net.neuron_catalogue = copy.deepcopy(neuron_catalogue)
        
        if not os.path.exists(net.features_file):
            raise FileNotFoundError(f"YAML file '{net.features_file}' not found")
        
        with open(net.features_file, "r") as f:
            net.features_dict = yaml.safe_load(f)

        net.population_rescale = 1.0 if "population_rescale_factor" not in net.features_dict else net.features_dict["population_rescale_factor"]
        net.connection_rescale = 1.0 if "connection_rescale_factor" not in net.features_dict else net.features_dict["connection_rescale_factor"]

        return net

    def build(self, progress_bar=None):
        if progress_bar is None:
            if spiking.VERBOSITY == 1:
                progress_bar = True
            else:
                progress_bar = False

        self.populations = dict()
        
        for pop in self.features_dict['populations']:
            features = self.features_dict['populations'][pop]
            paramap = self.neuron_catalogue[features['neuron_model']]
            try:
                self.populations[pop] = spiking.Population( int(self.population_rescale * features['size']), paramap, self._interface )
            except IndexError as e:
                message = f"While building population {pop} an error was raised:\n\t"
                message += str(e)
                raise IndexError(message)
        start = time()
        if "projections" in self.features_dict and self.features_dict['projections'] is not None:
            if progress_bar:
                iter = track(self.features_dict['projections'], description="Building connections..")
            else:
                iter = self.features_dict['projections']
            for proj in iter:
                features = self.features_dict['projections'][proj]
                try:
                    # Rescaling connections & weights
                    try:
                        features['connectivity'] *= self.connectivity_rescale_factor
                        features['weight'] *= self.weight_rescale_factor
                    except KeyError:
                        pass
                    try:
                        features['connectivity'] *= self.connectivity_rescale_factor
                        features['weight'] *= self.weight_rescale_factor
                    except KeyError:
                        pass

                    # Rescaling delays
                    try:
                        features['delay'] *= self.delay_rescale_factor
                    except KeyError:
                        pass

                    # Builds the projector
                    projector = spiking.SparseProjector(features, dist_type="lognorm")
                except ValueError as e:
                    raise ValueError(f"Some value was wrong during projection {efferent}->{afferent}")
                efferent, afferent = proj.split("->")
                efferent, afferent = efferent.strip(), afferent.strip()
                
                if efferent not in self.populations.keys():
                    raise KeyError(f"In projection {efferent} -> {afferent}: <{efferent}> was not defined")
                if afferent not in self.populations.keys():
                    raise KeyError(f"In projection {efferent} -> {afferent}: <{afferent}> was not defined")
                
                efferent = self.populations[efferent]
                afferent = self.populations[afferent]
                efferent.project(projector.get_projection(efferent, afferent), afferent)
        end = time()            
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

    def build(self, **params): 
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
        del params # Now use only self.params

        for param in self.params_value:
            for object in self.susceptibility_dict['parametric'][param]:
                # print(f"Parametrizing {object}")

                attribute = object['attribute']
                chi = object['susceptibility']
                parametric_delta = chi * (self.params_value[param] - self.params_shift[param])

                if "population" in object:
                    pop = object['population']

                    if pop not in self.features_dict['populations']:
                        raise KeyError(f"Population {pop} not found in network")
                    
                    is_pop_attr = False
                    is_neuron_attr = False

                    # Check if is a direct population attribute (size)
                    # I know it's uselessly too general, it's just in case I have add some pop attributes
                    try:
                        base_value = self.features_dict['populations'][pop][attribute]
                        self.features_dict['populations'][pop][attribute] = base_value + parametric_delta
                        is_pop_attr = True
                    except KeyError as error:
                        is_pop_attr = False

                    # Check if is a neuron attribute
                    try:
                        neuron_model = self.features_dict['populations'][pop]['neuron_model']
                        base_value = self.neuron_catalogue.neurons_dict[neuron_model][attribute]
                        self.neuron_catalogue.update(neuron_model, attribute, base_value + parametric_delta)
                        is_neuron_attr = True
                    except KeyError as e:
                        is_neuron_attr = False
                    
                    if not is_neuron_attr and not is_pop_attr:
                        raise KeyError(f"Paremetric attribute '{attribute}' was specified on population '{pop}' but was not found neither in the population nor in the neuron model.")

                elif "projection" in object:
                    proj = object['projection']
                    try:
                        _ =  self.features_dict['projections'][proj]
                    except KeyError as e:
                        raise KeyError(f"Projection '{proj}' was not found in {list(self.features_dict['projections'].keys())}")

                    try:
                        base_value = self.features_dict['projections'][proj][attribute]
                        self.features_dict['projections'][proj][attribute] = base_value + parametric_delta

                    except KeyError as e:
                        raise KeyError(f"Paremetric attribute '{attribute}' was specified on projection '{proj}' but was not found.")
                
                else:
                    raise KeyError(f"Parametric object {object} was not found in the network")

        super().build()


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
