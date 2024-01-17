"""A module for building objects from configuration files.
"""
import os
from time import time

import yaml
from rich import print
from rich.progress import track

import quilt.interface.spiking as spiking
import quilt.interface.base_objects as base_objects

class SpikingNetwork:

    def __init__(self):
        pass

    def run(self, dt=0.1, time=1):
        self.interface.run(dt, time)

    def rescale_populations(self, population_rescale_factor):
        self.population_rescale = population_rescale_factor
    
    def rescale_connectivity(self, connection_rescale_factor):
        self.connection_rescale = connection_rescale_factor

    @classmethod
    def from_yaml(cls, yaml_file, neuron_catalogue):
        net = cls()
        net.interface = spiking.SpikingNetwork("dummy")

        net.yaml_file = yaml_file
        net.neuron_catalogue = neuron_catalogue
        
        if not os.path.exists(net.yaml_file):
            raise FileNotFoundError("YAML file not found")
        
        with open(net.yaml_file, "r") as f:
            net.features_dict = yaml.safe_load(f)

        net.populations = dict()

        net.population_rescale = 1.0 if "population_rescale_factor" not in net.features_dict else net.features_dict["population_rescale_factor"]
        net.connection_rescale = 1.0 if "connection_rescale_factor" not in net.features_dict else net.features_dict["connection_rescale_factor"]

        return net
    
    def build(self):
        for pop in self.features_dict['populations']:
            paramap = self.neuron_catalogue[pop['neuron_model']]
            try:
                self.populations[pop['name']] = spiking.Population( int(self.population_rescale * pop['size']), paramap, self.interface )
            except IndexError as e:
                message = f"While building population {pop['name']} an error was raised:\n\t"
                message += str(e)
                raise IndexError(message)
        start = time()
        if "projections" in self.features_dict and self.features_dict['projections'] is not None:
            for proj in track(self.features_dict['projections'], description="Building connections.."):
                try:
                    # Rescaling connections
                    try:
                        proj['features']['inh_fraction'] *= self.connection_rescale
                    except KeyError:
                        pass
                    try:
                        proj['features']['exc_fraction'] *= self.connection_rescale
                    except KeyError:
                        pass

                    projector = spiking.RandomProjector(**(proj['features']))
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
            print(f"Loaded model for neuron '{neuron_name}'")
            catalogue.paramaps[neuron_name] = base_objects.ParaMap(catalogue.neurons_dict[neuron_name])
            catalogue.neuron_names += [neuron_name]
        
        return catalogue
    
    def __getitem__(self, neuron):
        try:
            paramap = self.paramaps[neuron]
        except KeyError:
            raise KeyError(f"Neural model '{neuron}' does not exist in this catalogue")

        return paramap
