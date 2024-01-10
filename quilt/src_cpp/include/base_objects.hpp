#pragma once
#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <map>

/**
 * @class HierarchicalID
 * @brief The ID object to identify nested structures
 * 
 * Create without arguments to get a supercontainer.
 * Then create a nested object by passing as argument 
 * the pointer to the parent container.
 * 
*/
class HierarchicalID{
    public:
        HierarchicalID * parent;

        HierarchicalID():parent(NULL),local_id(-1),n_subclasses(0){}
        HierarchicalID(HierarchicalID * parent): parent(parent),n_subclasses(0){
            this->local_id = this->parent->n_subclasses;
            this->parent->n_subclasses ++;
        }
        unsigned int get_id(){return this->local_id;}
    private:
        unsigned int local_id;
        unsigned int n_subclasses;
};

/**
 * @class EvolutuionContext
 * @brief The object that contains time evolution infos
 * 
 * @param[in] dt
 * 
*/
class EvolutionContext{
    public:
        double now; // time in millis
        double dt;  // timestep in millis

        EvolutionContext(double dt):dt(dt){
            this -> now = 0.0;
        }
        void do_step(){
            this -> now += this -> dt;
        }
};

/**
 * 
 * @class ParaMap
 * @brief a dictionary class
 * 
 * It's just a bridge between python and C++
 * 
*/
class ParaMap{
    public:
        std::map<std::string, float> value_map;

        ParaMap(){}
        ParaMap(const std::map<std::string, float> & value_map):value_map(value_map){}

        void add(const std::string& key, float value){value_map[key] = value;}
        float get(const std::string& key) const { return value_map.at(key);}
};