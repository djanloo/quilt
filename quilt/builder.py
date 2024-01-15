"""A module for building objects from configuration files.
"""
import os
import yaml

import quilt.bin.spiking as spiking

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
        
        for pop in net.features_dict['populations']:
            paramap = neuron_catalogue[pop['neuron_model']]
            net.populations[pop['name']] = spiking.Population(pop['size'], paramap, net.interface )
        
        if "projections" in net.features_dict and net.features_dict['projections'] is not None:
            for proj in net.features_dict['projections']:
                projector = spiking.RandomProjector(**(proj['features']))
                efferent = net.populations[proj['efferent']]
                afferent = net.populations[proj['afferent']]
                efferent.project(projector.get_projection(efferent, afferent), afferent)
            
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
            catalogue.paramaps[neuron_name] = spiking.ParaMap(catalogue.neurons_dict[neuron_name])
            catalogue.neuron_names += [neuron_name]
        
        return catalogue
    
    def __getitem__(self, neuron):
        try:
            paramap = self.paramaps[neuron]
        except KeyError:
            raise KeyError(f"Neural model '{neuron}' does not exist in this catalogue")

        return paramap
