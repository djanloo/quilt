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

    @classmethod
    def from_yaml(cls, yaml_file, neuron_catalogue):
        net = cls()
        net.interface = spiking.SpikingNetwork("dummy")

        net.yaml_file = yaml_file
        net.catalogue = neuron_catalogue
        
        if not os.path.exists(net.yaml_file):
            raise FileNotFoundError("YAML file not found")
        
        with open(net.yaml_file, "r") as f:
            net.features_dict = yaml.safe_load(f)

        net.populations = dict()

        population_rescale = 1.0 if "population_rescale_factor" not in net.features_dict else net.features_dict["population_rescale_factor"]
        connection_rescale = 1.0 if "connection_rescale_factor" not in net.features_dict else net.features_dict["connection_rescale_factor"]


        
        for pop in net.features_dict['populations']:
            paramap = neuron_catalogue[pop['neuron_model']]
            try:
                net.populations[pop['name']] = spiking.Population( int(population_rescale * pop['size']), paramap, net.interface )
            except IndexError as e:
                message = f"While building population {pop['name']} an error was raised:\n\t"
                message += str(e)
                raise IndexError(message)
        start = time()
        if "projections" in net.features_dict and net.features_dict['projections'] is not None:
            for proj in track(net.features_dict['projections'], description="Building connections.."):
                try:
                    # Rescaling connections
                    try:
                        proj['features']['inh_fraction'] *= connection_rescale
                    except KeyError:
                        pass
                    try:
                        proj['features']['exc_fraction'] *= connection_rescale
                    except KeyError:
                        pass

                    projector = spiking.RandomProjector(**(proj['features']))
                except ValueError as e:
                    raise ValueError(f"Some value was wrong during projection {efferent}->{afferent}")
                efferent, afferent = proj['name'].split("->")
                efferent, afferent = efferent.strip(), afferent.strip()
                
                if efferent not in net.populations.keys():
                    raise KeyError(f"In projection {efferent} -> {afferent}: <{efferent}> was not defined")
                if afferent not in net.populations.keys():
                    raise KeyError(f"In projection {efferent} -> {afferent}: <{afferent}> was not defined")
                
                efferent = net.populations[efferent]
                afferent = net.populations[afferent]
                efferent.project(projector.get_projection(efferent, afferent), afferent)
        end = time()            
        return net
    

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
