import os
import yaml

import quilt.bin.neur as neur

class SpikingNetwork:

    def __init__(self):
        pass

    @classmethod
    def from_yaml(cls, yaml_file):
        net = cls()
        net.interface = neur.SpikingNetwork("dummy")

        net.yaml_file = yaml_file
        
        if not os.path.exists(net.yaml_file):
            raise FileNotFoundError("YAML file not found")
        
        with open(net.yaml_file, "r") as f:
            net.features_dict = yaml.safe_load(f)

        net.populations = dict()
        
        for pop in net.features_dict['populations']:
            net.populations[pop['name']] = neur.Population(pop['size'], pop['neuron_type'], net.interface )

        for proj in net.features_dict['projections']:
            projector = neur.RandomProjector(**(proj['features']))
            efferent = net.populations[proj['efferent']]
            afferent = net.populations[proj['afferent']]
            efferent.project(projector.get_projection(efferent, afferent), afferent)
        
        return net