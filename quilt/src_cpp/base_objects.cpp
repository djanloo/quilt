#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <map>

#include "include/base_objects.hpp"

HierarchicalID::HierarchicalID(HierarchicalID * parent): parent(parent),n_subclasses(0){
    local_id = parent->n_subclasses;
    parent->n_subclasses ++;
}
unsigned int HierarchicalID::get_id(){return local_id;}

EvolutionContext::EvolutionContext(double dt):dt(dt),now(0.0){}
void EvolutionContext::do_step(){now += dt;}

ParaMap::ParaMap(const std::map<std::string, float> & value_map):value_map(value_map){}
void ParaMap::add(const std::string& key, float value){value_map[key] = value;}
float ParaMap::get(const std::string& key) const { return value_map.at(key);}
