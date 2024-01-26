"""A module for building objects from configuration files.
"""
import os
from time import time
import warnings

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
    def from_yaml(cls, yaml_file, neuron_catalogue):
        net = cls()
        net._interface = spiking.SpikingNetwork("dummy")

        net.yaml_file = yaml_file
        net.neuron_catalogue = neuron_catalogue
        
        if not os.path.exists(net.yaml_file):
            raise FileNotFoundError("YAML file not found")
        
        with open(net.yaml_file, "r") as f:
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
            paramap = self.neuron_catalogue[pop['neuron_model']]
            try:
                self.populations[pop['name']] = spiking.Population( int(self.population_rescale * pop['size']), paramap, self._interface )
            except IndexError as e:
                message = f"While building population {pop['name']} an error was raised:\n\t"
                message += str(e)
                raise IndexError(message)
        start = time()
        if "projections" in self.features_dict and self.features_dict['projections'] is not None:
            if progress_bar:
                iter = track(self.features_dict['projections'], description="Building connections..")
            else:
                iter = self.features_dict['projections']
            for proj in iter:
                try:
                    # Rescaling connections & weights
                    try:
                        proj['features']['connectivity'] *= self.connectivity_rescale_factor
                        proj['features']['weight'] *= self.weight_rescale_factor
                    except KeyError:
                        pass
                    try:
                        proj['features']['connectivity'] *= self.connectivity_rescale_factor
                        proj['features']['weight'] *= self.weight_rescale_factor
                    except KeyError:
                        pass

                    # Rescaling delays
                    try:
                        proj['features']['delay'] *= self.delay_rescale_factor
                    except KeyError:
                        pass

                    # Builds the projector
                    projector = spiking.SparseProjector(proj['features'], dist_type="lognorm")
                except ValueError as e:
                    raise ValueError(f"Some value was wrong during projection {efferent}->{afferent}")
                efferent, afferent = proj['name'].split("->")
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
    
    def __getitem__(self, neuron):
        try:
            paramap = self.paramaps[neuron]
        except KeyError:
            raise KeyError(f"Neural model '{neuron}' does not exist in this catalogue")

        return paramap
